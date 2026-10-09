"""读取并表示一个任意大小的 Agent 团队。

重要边界：
1. 这个文件定义“图应该怎样表示”，不写死任何五 Agent 结构。
2. 真正的 Agent、capability 和联络关系来自 YAML 配置。
3. TeamWorld 是后台真相。攻击者不能直接拿到这个对象。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import FeedbackLevel


@dataclass(frozen=True)
class AgentSpec:
    """一个 Agent 在后台配置中的完整定义。"""

    agent_id: str
    capabilities: frozenset[str]
    contacts: tuple[str, ...]
    role_id: str = "generic_agent"
    role_name: str = "Generic Agent"
    role_description: str = "Handles assigned team requests."


class TeamWorld:
    """保存完整联络图和 capability 分布。

    这个对象同时支持单 Agent 和多 Agent：
    - 单 Agent：只有一个节点，contacts 为空；
    - 多 Agent：配置任意多个节点和有向联络边。
    """

    def __init__(self, agents: Mapping[str, AgentSpec]) -> None:
        self._agents = dict(agents)
        self._validate()

    def _validate(self) -> None:
        if not self._agents:
            raise ValueError("场景至少需要一个 Agent。")

        for agent_id, spec in self._agents.items():
            if agent_id != spec.agent_id:
                raise ValueError(f"Agent 键 {agent_id!r} 与 agent_id 不一致。")
            if not agent_id.strip():
                raise ValueError("Agent 名字不能为空。")
            if not spec.role_id.strip():
                raise ValueError(f"Agent {agent_id!r} 的 role_id 不能为空。")
            if not spec.role_name.strip():
                raise ValueError(f"Agent {agent_id!r} 的 role_name 不能为空。")
            if not spec.role_description.strip():
                raise ValueError(
                    f"Agent {agent_id!r} 的 role_description 不能为空。"
                )
            if len(set(spec.contacts)) != len(spec.contacts):
                raise ValueError(f"Agent {agent_id!r} 的 contacts 存在重复项。")
            for contact in spec.contacts:
                if contact not in self._agents:
                    raise ValueError(
                        f"Agent {agent_id!r} 联系了不存在的 Agent {contact!r}。"
                    )

    @property
    def agent_ids(self) -> tuple[str, ...]:
        """后台使用的 Agent 名单；不要通过攻击者接口暴露。"""

        return tuple(self._agents)

    def has_agent(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def contacts_for(self, agent_id: str) -> tuple[str, ...]:
        """返回一个 Agent 的直接下游联系人。"""

        return self._get(agent_id).contacts

    def capabilities_for(self, agent_id: str) -> frozenset[str]:
        """返回一个 Agent 自己拥有的 capability。"""

        return self._get(agent_id).capabilities

    def spec_for(self, agent_id: str) -> AgentSpec:
        """返回一个 Agent 的后台完整配置。"""

        return self._get(agent_id)

    def role_id_for(self, agent_id: str) -> str:
        """返回一个 Agent 的角色模板 ID。"""

        return self._get(agent_id).role_id

    def role_name_for(self, agent_id: str) -> str:
        """返回一个 Agent 的英文角色名。"""

        return self._get(agent_id).role_name

    def role_description_for(self, agent_id: str) -> str:
        """返回一个 Agent 的英文职责描述。"""

        return self._get(agent_id).role_description

    def can_execute(self, agent_id: str, action: str) -> bool:
        """目标工具是否属于当前 Agent 的静态本地工具集合。"""

        return action in self.capabilities_for(agent_id)

    def _get(self, agent_id: str) -> AgentSpec:
        try:
            return self._agents[agent_id]
        except KeyError as exc:
            raise KeyError(f"未知 Agent：{agent_id!r}") from exc

    def shortest_path_to_capability(
        self,
        start_agent: str,
        action: str,
        *,
        max_hops: int | None = None,
    ) -> tuple[str, ...] | None:
        """后台计算到目标 capability 的最短路径。

        攻击者不能调用这个方法。评测器用它判断攻击找到的路径是不是最短。
        max_hops=None 表示不限制委托深度。
        """

        self._get(start_agent)
        queue: deque[tuple[str, tuple[str, ...]]] = deque(
            [(start_agent, (start_agent,))]
        )
        visited = {start_agent}

        while queue:
            current, path = queue.popleft()
            hops = len(path) - 1

            if self.can_execute(current, action):
                return path

            if max_hops is not None and hops >= max_hops:
                continue

            for contact in self.contacts_for(current):
                if contact in visited:
                    continue
                visited.add(contact)
                queue.append((contact, path + (contact,)))

        return None


@dataclass(frozen=True)
class Scenario:
    """一份完整实验配置。

    TeamWorld 描述团队；其余字段描述本次 episode 的目标和信息条件。
    """

    name: str
    world: TeamWorld
    start_agent: str
    target_action: str
    target_parameters: Mapping[str, Any]
    feedback_level: FeedbackLevel
    max_hops: int | None
    interaction_budget: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Scenario":
        """从 YAML 文件加载场景。"""

        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
        if not isinstance(raw, Mapping):
            raise ValueError("配置文件顶层必须是一个映射。")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Scenario":
        """从 Python 映射创建场景，方便测试和其他框架适配。"""

        scenario_raw = raw.get("scenario")
        agents_raw = raw.get("agents")
        if not isinstance(scenario_raw, Mapping):
            raise ValueError("配置缺少 scenario 段。")
        if not isinstance(agents_raw, Mapping):
            raise ValueError("配置缺少 agents 段。")

        agents: dict[str, AgentSpec] = {}
        for agent_id, spec_raw in agents_raw.items():
            if not isinstance(agent_id, str) or not isinstance(spec_raw, Mapping):
                raise ValueError("每个 Agent 都必须使用字符串名字和映射配置。")
            capabilities_raw = spec_raw.get("capabilities", [])
            contacts_raw = spec_raw.get("contacts", [])
            if not isinstance(capabilities_raw, list) or not all(
                isinstance(item, str) for item in capabilities_raw
            ):
                raise ValueError(f"Agent {agent_id!r} 的 capabilities 必须是字符串列表。")
            if not isinstance(contacts_raw, list) or not all(
                isinstance(item, str) for item in contacts_raw
            ):
                raise ValueError(f"Agent {agent_id!r} 的 contacts 必须是字符串列表。")
            agents[agent_id] = AgentSpec(
                agent_id=agent_id,
                capabilities=frozenset(capabilities_raw),
                contacts=tuple(contacts_raw),
                role_id=str(spec_raw.get("role_id", "generic_agent")),
                role_name=str(spec_raw.get("role_name", "Generic Agent")),
                role_description=str(
                    spec_raw.get(
                        "role_description",
                        "Handles assigned team requests.",
                    )
                ),
            )

        world = TeamWorld(agents)
        start_agent = str(scenario_raw.get("start_agent", ""))
        target_action = str(scenario_raw.get("target_action", ""))
        if not world.has_agent(start_agent):
            raise ValueError(f"start_agent {start_agent!r} 不在 agents 中。")
        if not target_action:
            raise ValueError("target_action 不能为空。")

        target_parameters = scenario_raw.get("target_parameters", {})
        if not isinstance(target_parameters, Mapping):
            raise ValueError("target_parameters 必须是一个映射。")

        max_hops_raw = scenario_raw.get("max_hops")
        if max_hops_raw is not None:
            if not isinstance(max_hops_raw, int) or max_hops_raw < 0:
                raise ValueError("max_hops 必须是非负整数或 null。")

        interaction_budget = scenario_raw.get("interaction_budget", 20)
        if not isinstance(interaction_budget, int) or interaction_budget <= 0:
            raise ValueError("interaction_budget 必须是正整数。")

        return cls(
            name=str(scenario_raw.get("name", "unnamed_scenario")),
            world=world,
            start_agent=start_agent,
            target_action=target_action,
            target_parameters=dict(target_parameters),
            feedback_level=FeedbackLevel(str(scenario_raw.get("feedback_level", "F2"))),
            max_hops=max_hops_raw,
            interaction_budget=interaction_budget,
        )
