"""GatePath 第一版实验使用的可复用有向图生成器。

本模块只负责拓扑，不负责攻击 prompt、AgentDojo 动作或模型调用。
所有图都使用不透露职责的 Agent ID；目标能力位置只存在于后台
``ExperimentTopology.target_agent`` 中，不应放进正常 Agent 的输入。

支持的图：

* ``G0``：仅入口节点；
* ``G(b,d)``：受控分叉/深度图，节点数 ``1 + b * d``；
* ``G(1,6)`` 与 ``G(6,1)``：七节点等规模形状对照；
* ``GDAG``：七节点稀疏有向无环图，包含分支和路线汇合。
* ``G32-X2/P2/X3/P3``：从 ``G(3,2)`` 派生的等边数多路径对照；
* ``G33-EDGE/CYCLE``：从 ``G(3,3)`` 派生的普通边/回边探索图；
* ``G33-X2/P2/X3/P3``：深度三的等边数多路径对照；
* ``G16-ACYCLIC-CONTROL/CYCLE``：深链上的可达无环/回边对照。
"""

from __future__ import annotations

import random
import re
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from .world import AgentSpec, TeamWorld


GRAPH_SCHEMA_VERSION = "0.3"
ENTRY_AGENT_ID = "Agent_A"
_CONTROLLED_NAME = re.compile(
    r"^g(?:\(|-b)(?P<branching>\d+)(?:,|-d)(?P<depth>\d+)\)?$",
    re.IGNORECASE,
)


class GraphFamily(str, Enum):
    """实验图在分析中的用途。"""

    DIRECT = "DIRECT"
    CORE = "CORE"
    EQUAL_SIZE = "EQUAL_SIZE"
    DAG = "DAG"
    MULTIPATH_CONTROL = "MULTIPATH_CONTROL"
    MULTIPATH = "MULTIPATH"
    CYCLE_CONTROL = "CYCLE_CONTROL"
    CYCLE = "CYCLE"


@dataclass(frozen=True, slots=True)
class ExperimentTopology:
    """一张与具体任务无关的完整后台拓扑。"""

    graph_id: str
    graph_name: str
    family: GraphFamily
    seed: int
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    entry_agent: str
    target_agent: str | None
    branching: int | None = None
    depth: int | None = None
    role_clone_pairs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.graph_id.strip():
            raise ValueError("graph_id 不能为空。")
        if not self.graph_name.strip():
            raise ValueError("graph_name 不能为空。")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("图中不能存在重复节点。")
        if not self.nodes:
            raise ValueError("图至少需要一个节点。")
        if self.entry_agent not in self.nodes:
            raise ValueError("entry_agent 必须属于图。")
        if self.target_agent is not None:
            if self.target_agent not in self.nodes:
                raise ValueError("target_agent 必须属于图。")
            if self.target_agent == self.entry_agent:
                raise ValueError("第一版目标 Agent 不能是入口 Agent。")
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("图中不能存在重复边。")
        for sender, receiver in self.edges:
            if sender not in self.nodes or receiver not in self.nodes:
                raise ValueError("边的两端都必须属于图。")
            if sender == receiver:
                raise ValueError("第一版图不允许自环。")
        for clone_id, reference_id in self.role_clone_pairs:
            if clone_id not in self.nodes or reference_id not in self.nodes:
                raise ValueError("role_clone_pairs中的Agent必须属于图。")
            if clone_id == reference_id:
                raise ValueError("Agent不能克隆自己的角色。")
            if clone_id in {self.entry_agent, self.target_agent}:
                raise ValueError("入口或目标Agent不能作为角色克隆节点。")
        if self.target_agent is not None:
            path = self.shortest_path_to_target()
            if path is None:
                raise ValueError("target_agent 必须能从 entry_agent 到达。")
            if self.depth is not None and len(path) - 1 != self.depth:
                raise ValueError("记录的 depth 与实际最短路径不一致。")

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def max_out_degree(self) -> int:
        return max(len(self.contacts_for(node)) for node in self.nodes)

    @property
    def is_dag(self) -> bool:
        indegree = {node: 0 for node in self.nodes}
        adjacency = {node: [] for node in self.nodes}
        for sender, receiver in self.edges:
            adjacency[sender].append(receiver)
            indegree[receiver] += 1

        queue = deque(node for node, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for receiver in adjacency[current]:
                indegree[receiver] -= 1
                if indegree[receiver] == 0:
                    queue.append(receiver)
        return visited == len(self.nodes)

    @property
    def has_cycle(self) -> bool:
        """图中是否至少存在一个有向环。"""

        return not self.is_dag

    def contacts_for(self, agent_id: str) -> tuple[str, ...]:
        if agent_id not in self.nodes:
            raise KeyError(f"未知 Agent：{agent_id!r}")
        return tuple(
            receiver for sender, receiver in self.edges if sender == agent_id
        )

    def shortest_path_to_target(self) -> tuple[str, ...] | None:
        """只供图构建器、运行器和评测器使用。"""

        if self.target_agent is None:
            return None
        queue: deque[tuple[str, tuple[str, ...]]] = deque(
            [(self.entry_agent, (self.entry_agent,))]
        )
        visited = {self.entry_agent}
        while queue:
            current, path = queue.popleft()
            if current == self.target_agent:
                return path
            for contact in self.contacts_for(current):
                if contact in visited:
                    continue
                visited.add(contact)
                queue.append((contact, path + (contact,)))
        return None

    def path_count_to_target(self) -> int:
        """返回从入口到目标的简单有向路径数量。

        含环图拥有无限多条允许重复节点的 walk，因此该指标明确只统计
        不重复访问节点的 simple path。第一版图最多十个节点，使用深度
        优先枚举既直观又足够快。
        """

        if self.target_agent is None:
            return 0

        def count_from(current: str, visited: frozenset[str]) -> int:
            if current == self.target_agent:
                return 1
            return sum(
                count_from(contact, visited | {contact})
                for contact in self.contacts_for(current)
                if contact not in visited
            )

        return count_from(self.entry_agent, frozenset({self.entry_agent}))

    def _topological_order(self) -> tuple[str, ...]:
        indegree = {node: 0 for node in self.nodes}
        for _, receiver in self.edges:
            indegree[receiver] += 1
        queue = deque(node for node in self.nodes if indegree[node] == 0)
        ordered: list[str] = []
        while queue:
            current = queue.popleft()
            ordered.append(current)
            for contact in self.contacts_for(current):
                indegree[contact] -= 1
                if indegree[contact] == 0:
                    queue.append(contact)
        if len(ordered) != len(self.nodes):
            raise RuntimeError("图不是 DAG。")
        return tuple(ordered)

    def to_world(
        self,
        target_action: str,
        *,
        non_target_capabilities: Mapping[str, Iterable[str]] | None = None,
    ) -> TeamWorld:
        """把拓扑绑定到一个目标动作，生成静态 per-agent 工具池。

        默认只有隐藏目标 Agent 拥有 ``target_action``。调用方可为其他
        Agent 注入不包含目标动作的普通工具，以便后续增加角色真实性。
        """

        if not isinstance(target_action, str) or not target_action.strip():
            raise ValueError("target_action 不能为空。")
        extra = non_target_capabilities or {}
        unknown = set(extra) - set(self.nodes)
        if unknown:
            raise ValueError(f"non_target_capabilities 包含未知 Agent：{unknown}")

        specs: dict[str, AgentSpec] = {}
        for agent_id in self.nodes:
            capabilities = set(extra.get(agent_id, ()))
            if (
                agent_id != self.target_agent
                and target_action in capabilities
            ):
                raise ValueError("目标动作只能分配给隐藏目标 Agent。")
            if agent_id == self.target_agent:
                capabilities.add(target_action)
            specs[agent_id] = AgentSpec(
                agent_id=agent_id,
                capabilities=frozenset(capabilities),
                contacts=self.contacts_for(agent_id),
            )
        return TeamWorld(specs)

    def as_dict(self) -> dict[str, Any]:
        """后台可复现实验配置；绝不能整体传给 Agent。"""

        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "graph_id": self.graph_id,
            "graph_name": self.graph_name,
            "family": self.family.value,
            "seed": self.seed,
            "entry_agent": self.entry_agent,
            "target_agent": self.target_agent,
            "branching": self.branching,
            "depth": self.depth,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "max_out_degree": self.max_out_degree,
            "is_dag": self.is_dag,
            "has_cycle": self.has_cycle,
            "nodes": list(self.nodes),
            "edges": [list(edge) for edge in self.edges],
            "shortest_target_path": (
                None
                if self.shortest_path_to_target() is None
                else list(self.shortest_path_to_target() or ())
            ),
            "target_path_count": self.path_count_to_target(),
            "role_clone_pairs": [
                [clone_id, reference_id]
                for clone_id, reference_id in self.role_clone_pairs
            ],
        }


def build_direct_topology() -> ExperimentTopology:
    """生成仅入口 Agent 的 G0，保留供以后对照使用。"""

    return ExperimentTopology(
        graph_id="g0",
        graph_name="G0",
        family=GraphFamily.DIRECT,
        seed=0,
        nodes=(ENTRY_AGENT_ID,),
        edges=(),
        entry_agent=ENTRY_AGENT_ID,
        target_agent=None,
        branching=0,
        depth=0,
    )


def build_controlled_topology(
    branching: int,
    depth: int,
    *,
    seed: int = 0,
) -> ExperimentTopology:
    """生成 ``N = 1 + branching * depth`` 的受控图。

    每一层只有上一层隐藏主路径节点继续分叉；其中一个子节点继续通往
    下一层，其余子节点成为死路。每层正确子节点及联系人顺序都由 seed
    决定，因此 Agent ID 或列表位置不会稳定泄露正确方向。
    """

    if branching <= 0:
        raise ValueError("branching 必须是正整数。")
    if depth <= 0:
        raise ValueError("depth 必须是正整数。")

    rng = random.Random(seed)
    nodes = [ENTRY_AGENT_ID]
    edges: list[tuple[str, str]] = []
    current = ENTRY_AGENT_ID

    for layer in range(1, depth + 1):
        children = [
            f"Agent_L{layer:02d}_{slot:02d}"
            for slot in range(1, branching + 1)
        ]
        nodes.extend(children)
        continuation = rng.choice(children)
        contact_order = list(children)
        rng.shuffle(contact_order)
        edges.extend((current, child) for child in contact_order)
        current = continuation

    family = (
        GraphFamily.CORE
        if branching in {1, 2, 3} and depth in {1, 2, 3}
        else GraphFamily.EQUAL_SIZE
    )
    return ExperimentTopology(
        graph_id=f"g-b{branching}-d{depth}-s{seed:04d}",
        graph_name=f"G({branching},{depth})",
        family=family,
        seed=seed,
        nodes=tuple(nodes),
        edges=tuple(edges),
        entry_agent=ENTRY_AGENT_ID,
        target_agent=current,
        branching=branching,
        depth=depth,
    )


def build_sparse_dag_topology(*, seed: int = 0) -> ExperimentTopology:
    """生成七节点 GDAG：两条路线在中间节点汇合后到达同一目标。"""

    rng = random.Random(seed)
    opaque = [f"Agent_{index:03d}" for index in range(1, 7)]
    rng.shuffle(opaque)
    branch_left, branch_right, merge, dead_left, dead_right, target = opaque

    left_contacts = [merge, dead_left]
    right_contacts = [merge, dead_right]
    entry_contacts = [branch_left, branch_right]
    rng.shuffle(left_contacts)
    rng.shuffle(right_contacts)
    rng.shuffle(entry_contacts)

    edges = [
        *( (ENTRY_AGENT_ID, node) for node in entry_contacts ),
        *( (branch_left, node) for node in left_contacts ),
        *( (branch_right, node) for node in right_contacts ),
        (merge, target),
    ]
    nodes = (ENTRY_AGENT_ID, *opaque)
    return ExperimentTopology(
        graph_id=f"gdag-s{seed:04d}",
        graph_name="GDAG",
        family=GraphFamily.DAG,
        seed=seed,
        nodes=nodes,
        edges=tuple(edges),
        entry_agent=ENTRY_AGENT_ID,
        target_agent=target,
        branching=None,
        depth=3,
    )


def build_g32_derived_topology(
    variant: str,
    *,
    seed: int = 0,
) -> ExperimentTopology:
    """从 ``G(3,2)`` 派生等节点、等深度的多路径/边数控制图。

    ``X2`` 与 ``P2`` 都只增加一条边，``X3`` 与 ``P3`` 都增加两条
    边。X 变体把新增路线接向死路，目标路径数仍为一；P 变体把同一
    起点接向隐藏目标，使目标路径数分别变为二和三。
    """

    normalized = variant.strip().upper()
    if normalized not in {"X2", "P2", "X3", "P3"}:
        raise ValueError("G(3,2) 派生图 variant 必须是 X2、P2、X3 或 P3。")

    base = build_controlled_topology(3, 2, seed=seed)
    path = base.shortest_path_to_target()
    if path is None:  # pragma: no cover - 基础构造器不可能产生此状态
        raise RuntimeError("G(3,2) 基础图缺少目标路径。")
    branch_agent, target = path[1], path[2]
    first_dead = tuple(
        node
        for node in base.contacts_for(base.entry_agent)
        if node != branch_agent
    )
    second_dead = tuple(
        node
        for node in base.contacts_for(branch_agent)
        if node != target
    )
    if len(first_dead) != 2 or len(second_dead) != 2:
        raise RuntimeError("G(3,2) 派生图需要每层恰好两个死路节点。")

    added_edge_count = 1 if normalized.endswith("2") else 2
    if normalized.startswith("X"):
        additions = tuple(
            (first_dead[index], second_dead[index])
            for index in range(added_edge_count)
        )
        family = GraphFamily.MULTIPATH_CONTROL
    else:
        additions = tuple(
            (first_dead[index], target)
            for index in range(added_edge_count)
        )
        family = GraphFamily.MULTIPATH

    return ExperimentTopology(
        graph_id=f"g32-{normalized.lower()}-s{seed:04d}",
        graph_name=f"G32-{normalized}",
        family=family,
        seed=seed,
        nodes=base.nodes,
        edges=(*base.edges, *additions),
        entry_agent=base.entry_agent,
        target_agent=base.target_agent,
        branching=base.branching,
        depth=base.depth,
    )


def build_g33_derived_topology(
    variant: str,
    *,
    seed: int = 0,
) -> ExperimentTopology:
    """从 ``G(3,3)`` 派生普通边/回边配对图。

    两个变体使用相同的边起点并各增加一条边。``EDGE`` 把第三层的
    一个死路连向同层另一死路，保持无环；``CYCLE`` 将它连回第一层
    主路径节点，形成一个不改变最短目标深度的有向环。
    """

    normalized = variant.strip().upper()
    if normalized not in {"EDGE", "CYCLE"}:
        raise ValueError("G(3,3) 派生图 variant 必须是 EDGE 或 CYCLE。")

    base = build_controlled_topology(3, 3, seed=seed)
    path = base.shortest_path_to_target()
    if path is None:  # pragma: no cover - 基础构造器不可能产生此状态
        raise RuntimeError("G(3,3) 基础图缺少目标路径。")
    first_path_agent, final_parent, target = path[1], path[2], path[3]
    final_dead = tuple(
        node
        for node in base.contacts_for(final_parent)
        if node != target
    )
    if len(final_dead) != 2:
        raise RuntimeError("G(3,3) 派生图需要第三层恰好两个死路节点。")

    if normalized == "EDGE":
        addition = (final_dead[0], final_dead[1])
        family = GraphFamily.CYCLE_CONTROL
    else:
        addition = (final_dead[0], first_path_agent)
        family = GraphFamily.CYCLE

    return ExperimentTopology(
        graph_id=f"g33-{normalized.lower()}-s{seed:04d}",
        graph_name=f"G33-{normalized}",
        family=family,
        seed=seed,
        nodes=base.nodes,
        edges=(*base.edges, addition),
        entry_agent=base.entry_agent,
        target_agent=base.target_agent,
        branching=base.branching,
        depth=base.depth,
    )


def build_g33_multipath_topology(
    variant: str,
    *,
    seed: int = 0,
) -> ExperimentTopology:
    """从 ``G(3,3)`` 派生深度三的等边数多路径对照。

    P变体把第一层死路接到第二层主路径节点，从而新增长度仍为三的
    目标路径。X变体使用相同起点和边数，但接到不能到达目标的第二层
    死路。节点、目标Agent和最短深度都保持不变。
    """

    normalized = variant.strip().upper()
    if normalized not in {"X2", "P2", "X3", "P3"}:
        raise ValueError("G(3,3) 多路径 variant 必须是 X2、P2、X3 或 P3。")

    base = build_controlled_topology(3, 3, seed=seed)
    path = base.shortest_path_to_target()
    if path is None:  # pragma: no cover - 基础构造器不可能产生此状态
        raise RuntimeError("G(3,3) 基础图缺少目标路径。")
    first_path_agent, second_path_agent = path[1], path[2]
    first_dead = tuple(
        node
        for node in base.contacts_for(base.entry_agent)
        if node != first_path_agent
    )
    second_dead = tuple(
        node
        for node in base.contacts_for(first_path_agent)
        if node != second_path_agent
    )
    if len(first_dead) != 2 or len(second_dead) != 2:
        raise RuntimeError("G(3,3) 多路径图需要前两层各有两个死路节点。")

    added_edge_count = 1 if normalized.endswith("2") else 2
    if normalized.startswith("X"):
        additions = tuple(
            (first_dead[index], second_dead[index])
            for index in range(added_edge_count)
        )
        family = GraphFamily.MULTIPATH_CONTROL
    else:
        additions = tuple(
            (first_dead[index], second_path_agent)
            for index in range(added_edge_count)
        )
        family = GraphFamily.MULTIPATH

    return ExperimentTopology(
        graph_id=f"g33-{normalized.lower()}-s{seed:04d}",
        graph_name=f"G33-{normalized}",
        family=family,
        seed=seed,
        nodes=base.nodes,
        edges=(*base.edges, *additions),
        entry_agent=base.entry_agent,
        target_agent=base.target_agent,
        branching=base.branching,
        depth=base.depth,
    )


def build_g16_cycle_topology(
    variant: str,
    *,
    seed: int = 0,
) -> ExperimentTopology:
    """建立深链上可达的无环新节点/回访旧节点配对图。

    两图都从 ``G(1,6)`` 增加一个辅助Agent，并让深度二节点多出一个
    联系人。无环图联系新的辅助Agent；环路图联系角色与辅助Agent严格
    匹配、但已经访问过的深度一Agent。这样Agent看到的公开角色线索
    一致，关键区别是接收者是新Agent还是拥有会话历史的旧Agent。
    """

    normalized = variant.strip().upper().replace("_", "-")
    if normalized not in {"ACYCLIC-CONTROL", "CYCLE"}:
        raise ValueError(
            "G(1,6) 环路 variant 必须是 ACYCLIC-CONTROL 或 CYCLE。"
        )

    base = build_controlled_topology(1, 6, seed=seed)
    path = base.shortest_path_to_target()
    if path is None:  # pragma: no cover - 基础构造器不可能产生此状态
        raise RuntimeError("G(1,6) 基础图缺少目标路径。")
    first_agent, second_agent = path[1], path[2]
    fresh_agent = "Agent_LOOP_FRESH"
    receiver = fresh_agent if normalized == "ACYCLIC-CONTROL" else first_agent
    family = (
        GraphFamily.CYCLE_CONTROL
        if normalized == "ACYCLIC-CONTROL"
        else GraphFamily.CYCLE
    )

    return ExperimentTopology(
        graph_id=f"g16-{normalized.lower()}-s{seed:04d}",
        graph_name=f"G16-{normalized}",
        family=family,
        seed=seed,
        nodes=(*base.nodes, fresh_agent),
        edges=(*base.edges, (second_agent, receiver)),
        entry_agent=base.entry_agent,
        target_agent=base.target_agent,
        branching=base.branching,
        depth=base.depth,
        role_clone_pairs=((fresh_agent, first_agent),),
    )


def build_experiment_topologies(
    *,
    seed: int = 0,
    include_direct: bool = True,
) -> tuple[ExperimentTopology, ...]:
    """返回24张已实现实验图，并可选保留不进入主矩阵的 G0。"""

    graphs: list[ExperimentTopology] = []
    if include_direct:
        graphs.append(build_direct_topology())
    graphs.extend(
        build_controlled_topology(branching, depth, seed=seed)
        for branching in (1, 2, 3)
        for depth in (1, 2, 3)
    )
    graphs.extend(
        (
            build_controlled_topology(1, 6, seed=seed),
            build_controlled_topology(6, 1, seed=seed),
            build_sparse_dag_topology(seed=seed),
            build_g32_derived_topology("X2", seed=seed),
            build_g32_derived_topology("P2", seed=seed),
            build_g32_derived_topology("X3", seed=seed),
            build_g32_derived_topology("P3", seed=seed),
            build_g33_derived_topology("EDGE", seed=seed),
            build_g33_derived_topology("CYCLE", seed=seed),
            build_g33_multipath_topology("X2", seed=seed),
            build_g33_multipath_topology("P2", seed=seed),
            build_g33_multipath_topology("X3", seed=seed),
            build_g33_multipath_topology("P3", seed=seed),
            build_g16_cycle_topology("ACYCLIC-CONTROL", seed=seed),
            build_g16_cycle_topology("CYCLE", seed=seed),
        )
    )
    return tuple(graphs)


def build_named_topology(
    name: str,
    *,
    seed: int = 0,
) -> ExperimentTopology:
    """根据 CLI/配置中的短名字创建图。

    接受 ``G0``、``GDAG``、``G(2,3)``、``g-b2-d3`` 及派生图名。
    """

    normalized = name.strip().lower().replace(" ", "")
    if normalized == "g0":
        return build_direct_topology()
    if normalized == "gdag":
        return build_sparse_dag_topology(seed=seed)
    g32_variants = {
        "g32-x2": "X2",
        "g32-p2": "P2",
        "g32-x3": "X3",
        "g32-p3": "P3",
    }
    if normalized in g32_variants:
        return build_g32_derived_topology(
            g32_variants[normalized],
            seed=seed,
        )
    g33_variants = {
        "g33-edge": "EDGE",
        "g33-cycle": "CYCLE",
    }
    if normalized in g33_variants:
        return build_g33_derived_topology(
            g33_variants[normalized],
            seed=seed,
        )
    g33_multipath_variants = {
        "g33-x2": "X2",
        "g33-p2": "P2",
        "g33-x3": "X3",
        "g33-p3": "P3",
    }
    if normalized in g33_multipath_variants:
        return build_g33_multipath_topology(
            g33_multipath_variants[normalized],
            seed=seed,
        )
    g16_variants = {
        "g16-acyclic-control": "ACYCLIC-CONTROL",
        "g16-cycle": "CYCLE",
    }
    if normalized in g16_variants:
        return build_g16_cycle_topology(
            g16_variants[normalized],
            seed=seed,
        )
    match = _CONTROLLED_NAME.fullmatch(normalized)
    if match is None:
        raise ValueError(
            "未知图名字；使用 G0、GDAG、G(b,d)、g-b< b >-d< d >、"
            "G32-X2/P2/X3/P3、G33-EDGE/CYCLE、"
            "G33-X2/P2/X3/P3 或 G16-ACYCLIC-CONTROL/CYCLE。"
        )
    return build_controlled_topology(
        int(match.group("branching")),
        int(match.group("depth")),
        seed=seed,
    )
