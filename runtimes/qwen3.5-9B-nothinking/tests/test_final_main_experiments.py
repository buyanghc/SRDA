from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import yaml

from gatepath.experiment_graphs import build_named_topology
from gatepath.target_instances import load_formal_target_instances


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
SELECTION_PATH = (
    PROJECT_ROOT
    / "gatepath"
    / "data"
    / "formal_target_selection_3_per_type_v1.yaml"
)

FINAL_CONFIG_NAMES = (
    "request_response_final_system_terminal_full.yaml",
    "request_response_final_system_empty_full.yaml",
    "request_response_final_system_natural_full.yaml",
    "request_response_final_forward_terminal_full.yaml",
    "request_response_final_forward_empty_full.yaml",
    "request_response_final_forward_natural_full.yaml",
)


def _load(name: str) -> dict:
    raw = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def test_final_target_selection_has_three_instances_per_capability() -> None:
    selection = yaml.safe_load(SELECTION_PATH.read_text(encoding="utf-8"))
    registry = {
        instance.instance_id: instance
        for instance in load_formal_target_instances()
    }
    selected = [registry[item] for item in selection["instance_ids"]]

    assert selection["status"] == "frozen"
    assert len(selected) == 45
    assert len({item.instance_id for item in selected}) == 45
    assert Counter(item.target_type_id for item in selected) == {
        **{f"W{index:02d}": 3 for index in range(1, 11)},
        **{f"B{index:02d}": 3 for index in range(1, 6)},
    }
    assert Counter(item.suite_name for item in selected) == {
        "workspace": 30,
        "banking": 15,
    }


def test_final_six_condition_matrix_is_frozen_and_paired() -> None:
    configs = [_load(name) for name in FINAL_CONFIG_NAMES]
    selection_sha = hashlib.sha256(SELECTION_PATH.read_bytes()).hexdigest()
    expected_graphs = (
        "G(1,2)",
        "G(1,3)",
        "G(2,2)",
        "G(2,3)",
        "G(3,2)",
        "G(3,3)",
        "G(1,6)",
        "GDAG",
        "G33-X2",
        "G33-P2",
        "G33-X3",
        "G33-P3",
        "G16-ACYCLIC-CONTROL",
        "G16-CYCLE",
    )

    for config in configs:
        assert config["status"] == "frozen"
        assert config["budget"] == {
            "max_tool_iterations": 4,
            "max_messages": 16,
            "max_runtime_seconds": 90,
        }
        assert config["config_id"].endswith("-bounded-20260820")
        assert config["supersedes"].endswith("-20260820")
        assert tuple(config["graphs"]["names"]) == expected_graphs
        assert all(
            len(build_named_topology(name, seed=0).shortest_path_to_target())
            - 1
            > 1
            for name in expected_graphs
        )
        assert len(config["randomization"]["seed_pairs"]) == 3
        assert config["targets"]["selection"] == (
            "fixed_selection_manifest"
        )
        assert config["targets"]["expected_registry_instance_count"] == 60
        assert config["targets"]["expected_instance_count"] == 45
        assert config["targets"]["selection_manifest_sha256"] == (
            selection_sha
        )
        assert config["experiment"]["canonical_work_item_policy"] == (
            "runtime_preserved_at_every_handoff"
        )
        assert config["recording"]["record_elapsed_seconds"] is True
        assert config["recording"]["record_model_token_usage"] is True

    first = configs[0]
    for config in configs[1:]:
        assert config["budget"] == first["budget"]
        assert config["randomization"] == first["randomization"]
        assert config["targets"] == first["targets"]
        assert config["graphs"] == first["graphs"]
        assert config["success_definition"] == first["success_definition"]
        assert config["recording"] == first["recording"]


def test_final_feedback_regimes_and_task_prompts_are_factorized() -> None:
    configs = {name: _load(name) for name in FINAL_CONFIG_NAMES}
    system = {
        regime: configs[f"request_response_final_system_{regime}_full.yaml"]
        for regime in ("terminal", "empty", "natural")
    }
    forward = {
        regime: configs[f"request_response_final_forward_{regime}_full.yaml"]
        for regime in ("terminal", "empty", "natural")
    }

    for family in (system, forward):
        assert family["terminal"]["experiment"]["response_visibility"] == (
            "hidden"
        )
        assert family["terminal"]["experiment"]["handoff_continuation"] == (
            "terminal_after_send"
        )
        assert family["terminal"]["experiment"][
            "downstream_response_policy"
        ] == "record_response_end_sender_turn"
        assert family["empty"]["experiment"]["response_visibility"] == (
            "hidden"
        )
        assert family["empty"]["experiment"]["handoff_continuation"] == (
            "return_to_sender"
        )
        assert family["natural"]["experiment"]["response_visibility"] == (
            "natural"
        )
        assert family["natural"]["experiment"]["handoff_continuation"] == (
            "return_to_sender"
        )

    system_versions = {
        config["model"]["prompt_version"] for config in system.values()
    }
    forward_versions = {
        config["model"]["prompt_version"] for config in forward.values()
    }
    assert system_versions == {
        "gatepath-request-response-system-handoff-baseline-20260818-en"
    }
    assert forward_versions == {
        "gatepath-request-response-system-handoff-forward-20260818-en"
    }

    for regime in ("terminal", "empty", "natural"):
        assert system[regime]["experiment"]["downstream_request_policy"] == (
            "system_handoff_only"
        )
        assert forward[regime]["experiment"][
            "downstream_request_policy"
        ] == "system_handoff_plus_task_forward"



def test_final_matrix_has_11340_formal_episodes() -> None:
    per_condition = 45 * 14 * 3
    assert per_condition == 1890
    assert len(FINAL_CONFIG_NAMES) * per_condition == 11340
