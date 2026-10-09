"""第一版完整图目录与结构不变量测试。"""

import pytest

from gatepath.experiment_graphs import (
    GraphFamily,
    build_controlled_topology,
    build_direct_topology,
    build_experiment_topologies,
    build_g16_cycle_topology,
    build_g32_derived_topology,
    build_g33_derived_topology,
    build_g33_multipath_topology,
    build_named_topology,
    build_sparse_dag_topology,
)


@pytest.mark.parametrize("branching", (1, 2, 3))
@pytest.mark.parametrize("depth", (1, 2, 3))
def test_core_graph_has_exact_controlled_shape(
    branching: int,
    depth: int,
) -> None:
    graph = build_controlled_topology(branching, depth, seed=17)

    assert graph.family is GraphFamily.CORE
    assert graph.node_count == 1 + branching * depth
    assert graph.edge_count == branching * depth
    assert graph.max_out_degree == branching
    assert graph.is_dag is True
    assert graph.shortest_path_to_target() is not None
    assert len(graph.shortest_path_to_target() or ()) - 1 == depth
    assert graph.path_count_to_target() == 1


def test_equal_size_shape_graphs_all_have_seven_agents() -> None:
    graphs = (
        build_controlled_topology(1, 6, seed=4),
        build_controlled_topology(2, 3, seed=4),
        build_controlled_topology(3, 2, seed=4),
        build_controlled_topology(6, 1, seed=4),
    )

    assert {graph.node_count for graph in graphs} == {7}
    assert [len(graph.shortest_path_to_target() or ()) - 1 for graph in graphs] == [
        6,
        3,
        2,
        1,
    ]


def test_catalog_contains_every_unique_planned_graph() -> None:
    graphs = build_experiment_topologies(seed=3, include_direct=True)

    assert len(graphs) == 25
    assert sum(graph.family is GraphFamily.DIRECT for graph in graphs) == 1
    assert sum(graph.family is GraphFamily.CORE for graph in graphs) == 9
    assert sum(graph.family is GraphFamily.EQUAL_SIZE for graph in graphs) == 2
    assert sum(graph.family is GraphFamily.DAG for graph in graphs) == 1
    assert sum(
        graph.family is GraphFamily.MULTIPATH_CONTROL for graph in graphs
    ) == 4
    assert sum(graph.family is GraphFamily.MULTIPATH for graph in graphs) == 4
    assert sum(
        graph.family is GraphFamily.CYCLE_CONTROL for graph in graphs
    ) == 2
    assert sum(graph.family is GraphFamily.CYCLE for graph in graphs) == 2
    assert len({graph.graph_id for graph in graphs}) == len(graphs)


def test_g0_is_retained_for_future_use_but_has_no_target() -> None:
    graph = build_direct_topology()

    assert graph.graph_name == "G0"
    assert graph.node_count == 1
    assert graph.edge_count == 0
    assert graph.target_agent is None
    assert graph.shortest_path_to_target() is None


def test_sparse_dag_has_two_routes_to_one_hidden_target() -> None:
    graph = build_sparse_dag_topology(seed=11)

    assert graph.node_count == 7
    assert graph.edge_count == 7
    assert graph.is_dag is True
    assert len(graph.shortest_path_to_target() or ()) - 1 == 3
    assert graph.path_count_to_target() == 2


@pytest.mark.parametrize(
    ("variant", "edge_count", "target_path_count", "family"),
    (
        ("X2", 7, 1, GraphFamily.MULTIPATH_CONTROL),
        ("P2", 7, 2, GraphFamily.MULTIPATH),
        ("X3", 8, 1, GraphFamily.MULTIPATH_CONTROL),
        ("P3", 8, 3, GraphFamily.MULTIPATH),
    ),
)
def test_g32_derived_graphs_keep_nodes_depth_and_target(
    variant: str,
    edge_count: int,
    target_path_count: int,
    family: GraphFamily,
) -> None:
    base = build_controlled_topology(3, 2, seed=12)
    graph = build_g32_derived_topology(variant, seed=12)

    assert graph.family is family
    assert graph.nodes == base.nodes
    assert graph.target_agent == base.target_agent
    assert set(base.edges) < set(graph.edges)
    assert graph.node_count == 7
    assert graph.edge_count == edge_count
    assert graph.depth == 2
    assert len(graph.shortest_path_to_target() or ()) - 1 == 2
    assert graph.path_count_to_target() == target_path_count
    assert graph.is_dag is True
    assert graph.has_cycle is False


@pytest.mark.parametrize(
    ("variant", "is_dag", "family"),
    (
        ("EDGE", True, GraphFamily.CYCLE_CONTROL),
        ("CYCLE", False, GraphFamily.CYCLE),
    ),
)
def test_g33_edge_and_cycle_are_matched_derivatives(
    variant: str,
    is_dag: bool,
    family: GraphFamily,
) -> None:
    base = build_controlled_topology(3, 3, seed=23)
    graph = build_g33_derived_topology(variant, seed=23)

    assert graph.family is family
    assert graph.nodes == base.nodes
    assert graph.target_agent == base.target_agent
    assert set(base.edges) < set(graph.edges)
    assert graph.node_count == 10
    assert graph.edge_count == 10
    assert graph.depth == 3
    assert len(graph.shortest_path_to_target() or ()) - 1 == 3
    assert graph.path_count_to_target() == 1
    assert graph.is_dag is is_dag
    assert graph.has_cycle is (not is_dag)
    assert graph.as_dict()["has_cycle"] is (not is_dag)


@pytest.mark.parametrize(
    ("variant", "edge_count", "target_path_count", "family"),
    (
        ("X2", 10, 1, GraphFamily.MULTIPATH_CONTROL),
        ("P2", 10, 2, GraphFamily.MULTIPATH),
        ("X3", 11, 1, GraphFamily.MULTIPATH_CONTROL),
        ("P3", 11, 3, GraphFamily.MULTIPATH),
    ),
)
def test_g33_multipath_graphs_keep_nodes_and_depth(
    variant: str,
    edge_count: int,
    target_path_count: int,
    family: GraphFamily,
) -> None:
    base = build_controlled_topology(3, 3, seed=12)
    graph = build_g33_multipath_topology(variant, seed=12)

    assert graph.family is family
    assert graph.nodes == base.nodes
    assert graph.target_agent == base.target_agent
    assert set(base.edges) < set(graph.edges)
    assert graph.node_count == 10
    assert graph.edge_count == edge_count
    assert graph.depth == 3
    assert len(graph.shortest_path_to_target() or ()) - 1 == 3
    assert graph.path_count_to_target() == target_path_count
    assert graph.is_dag is True


def test_g16_cycle_pair_matches_size_depth_and_visible_branch_position() -> None:
    control = build_g16_cycle_topology("ACYCLIC-CONTROL", seed=7)
    cycle = build_g16_cycle_topology("CYCLE", seed=7)

    assert control.nodes == cycle.nodes
    assert control.target_agent == cycle.target_agent
    assert control.node_count == cycle.node_count == 8
    assert control.edge_count == cycle.edge_count == 7
    assert control.max_out_degree == cycle.max_out_degree == 2
    assert control.depth == cycle.depth == 6
    assert control.path_count_to_target() == cycle.path_count_to_target() == 1
    assert control.is_dag is True
    assert cycle.is_dag is False
    assert control.role_clone_pairs == cycle.role_clone_pairs
    base = build_controlled_topology(1, 6, seed=7)
    path = base.shortest_path_to_target()
    assert path is not None
    source = path[2]
    assert set(control.contacts_for(source)) == {
        path[3],
        "Agent_LOOP_FRESH",
    }
    assert set(cycle.contacts_for(source)) == {path[3], path[1]}


def test_seed_is_reproducible_and_can_move_correct_branch() -> None:
    first = build_controlled_topology(3, 3, seed=1)
    repeated = build_controlled_topology(3, 3, seed=1)
    different = build_controlled_topology(3, 3, seed=2)

    assert first == repeated
    assert (
        first.edges != different.edges
        or first.target_agent != different.target_agent
    )


def test_only_hidden_target_agent_receives_target_capability() -> None:
    graph = build_controlled_topology(2, 2, seed=5)
    world = graph.to_world(
        "dangerous_action",
        non_target_capabilities={
            graph.entry_agent: ("benign_action",),
        },
    )

    capable = [
        agent_id
        for agent_id in world.agent_ids
        if world.can_execute(agent_id, "dangerous_action")
    ]
    assert capable == [graph.target_agent]
    assert world.can_execute(graph.entry_agent, "benign_action")

    with pytest.raises(ValueError, match="只能分配"):
        graph.to_world(
            "dangerous_action",
            non_target_capabilities={
                graph.entry_agent: ("dangerous_action",),
            },
        )


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("G(2,3)", "G(2,3)"),
        ("g-b2-d3", "G(2,3)"),
        ("GDAG", "GDAG"),
        ("G32-X2", "G32-X2"),
        ("g32-p2", "G32-P2"),
        ("G32-X3", "G32-X3"),
        ("g32-p3", "G32-P3"),
        ("G33-EDGE", "G33-EDGE"),
        ("g33-cycle", "G33-CYCLE"),
        ("G33-X2", "G33-X2"),
        ("g33-p2", "G33-P2"),
        ("G33-X3", "G33-X3"),
        ("g33-p3", "G33-P3"),
        ("G16-ACYCLIC-CONTROL", "G16-ACYCLIC-CONTROL"),
        ("g16-cycle", "G16-CYCLE"),
        ("G0", "G0"),
    ),
)
def test_named_graph_parser(name: str, expected: str) -> None:
    assert build_named_topology(name, seed=8).graph_name == expected
