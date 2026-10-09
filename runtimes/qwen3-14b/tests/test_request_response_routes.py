"""多路径与环路的可信后台统计测试。"""

from gatepath.request_response_protocol import DirectTeamRequest
from gatepath.request_response_runtime import summarize_request_routes


def test_route_summary_counts_revisits_repeated_edges_and_cycle_closures() -> None:
    requests = {
        "R1": DirectTeamRequest(
            root_request_id="ROOT",
            request_id="R1",
            parent_request_id=None,
            sender_id="Agent_A",
            receiver_id="Agent_B",
            content="work",
        ),
        "R2": DirectTeamRequest(
            root_request_id="ROOT",
            request_id="R2",
            parent_request_id="R1",
            sender_id="Agent_B",
            receiver_id="Agent_C",
            content="work",
        ),
        "R3": DirectTeamRequest(
            root_request_id="ROOT",
            request_id="R3",
            parent_request_id="R2",
            sender_id="Agent_C",
            receiver_id="Agent_B",
            content="work",
        ),
        "R4": DirectTeamRequest(
            root_request_id="ROOT",
            request_id="R4",
            parent_request_id="R3",
            sender_id="Agent_B",
            receiver_id="Agent_C",
            content="work",
        ),
    }

    summary = summarize_request_routes(requests)

    assert summary["node_visit_counts"] == {
        "Agent_B": 2,
        "Agent_C": 2,
    }
    assert summary["repeated_node_visit_count"] == 2
    assert summary["repeated_edge_traversal_count"] == 1
    assert summary["cycle_closing_message_count"] == 2
    assert summary["cycle_closing_request_ids"] == ["R3", "R4"]
