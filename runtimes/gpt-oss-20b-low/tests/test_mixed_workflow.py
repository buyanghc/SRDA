from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import yaml

from gatepath.mixed_workflow import run_mixed_workflow_episode_async
from gatepath.mixed_workflow_batch import (
    MixedWorkflowBatchStore,
    build_mixed_workflow_episode_specs,
)
from gatepath.request_response_protocol import (
    HandoffContinuation,
    REQUEST_RESPONSE_REVISED_BASELINE_PROMPT_VERSION,
    REQUEST_RESPONSE_REVISED_FORWARD_PROMPT_VERSION,
    RequestResponseBudget,
    ResponseVisibility,
    get_request_response_prompt_bundle,
)
from gatepath.workflow_graphs import workflow_graph_by_id
from gatepath.workflow_scenarios import workflow_case_by_id
from scripts.run_mixed_workflow_batch import _validate_prompt_policies


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    return yaml.safe_load(
        (
            PROJECT_ROOT
            / "configs"
            / "semantic_workflow_mixed_full.yaml"
        ).read_text(encoding="utf-8")
    )


def _build_specs() -> tuple:
    config = _load_config()
    selections = tuple(
        (
            tuple(config[domain]["graph_ids"]),
            tuple(config[domain]["normal_case_pair_ids"]),
            dict(config[domain]["attack_pair_by_normal_pair"]),
        )
        for domain in config["experiment"]["domains"]
    )
    return build_mixed_workflow_episode_specs(
        domain_selections=selections,
        conditions=tuple(config["experiment"]["conditions"]),
        regimes=tuple(config["regimes"]),
        repetitions=int(config["experiment"]["repetitions"]),
    )


def test_mixed_workflow_full_matrix_adds_108_attacked_episodes() -> None:
    specs = _build_specs()
    assert len(specs) == 108
    assert len({spec.episode_key for spec in specs}) == 108
    assert all(spec.condition == "normal_plus_attack" for spec in specs)
    assert sum(spec.regime == "open_loop" for spec in specs) == 54
    assert sum(spec.regime == "natural_feedback" for spec in specs) == 54
    for spec in specs:
        normal = workflow_case_by_id(spec.normal_pair_id).target_instance()
        attack = workflow_case_by_id(spec.attack_pair_id).target_instance()
        assert normal.suite_name == attack.suite_name
        assert normal.action == attack.action
        assert dict(normal.parameters) != dict(attack.parameters)




def test_system_only_full_mixed_config_has_108_episodes() -> None:
    expected_sizes = {
        "semantic_workflow_system_only_mixed_full.yaml": 108,
    }
    for filename, expected_size in expected_sizes.items():
        config = yaml.safe_load(
            (PROJECT_ROOT / "configs" / filename).read_text(encoding="utf-8")
        )
        selections = tuple(
            (
                tuple(config[domain]["graph_ids"]),
                tuple(config[domain]["normal_case_pair_ids"]),
                dict(config[domain]["attack_pair_by_normal_pair"]),
            )
            for domain in config["experiment"]["domains"]
        )
        specs = build_mixed_workflow_episode_specs(
            domain_selections=selections,
            conditions=tuple(config["experiment"]["conditions"]),
            regimes=tuple(config["regimes"]),
            repetitions=int(config["experiment"]["repetitions"]),
        )
        assert len(specs) == expected_size
        assert config["experiment"]["attack_task_carried_routing_policy"] == "none"
        assert config["model"]["prompt_versions"]["legitimate"] == config["model"]["prompt_versions"]["attack"]


def test_prompt_policy_validation_accepts_forward_and_system_only_attacks() -> None:
    baseline = get_request_response_prompt_bundle(
        "en",
        version=f"{REQUEST_RESPONSE_REVISED_BASELINE_PROMPT_VERSION}-en",
    )
    forward = get_request_response_prompt_bundle(
        "en",
        version=f"{REQUEST_RESPONSE_REVISED_FORWARD_PROMPT_VERSION}-en",
    )
    _validate_prompt_policies(
        experiment={
            "legitimate_task_carried_routing_policy": "none",
            "attack_task_carried_routing_policy": "exact_revised_forward_instruction",
        },
        legitimate_prompts=baseline,
        attack_prompts=forward,
    )
    _validate_prompt_policies(
        experiment={
            "legitimate_task_carried_routing_policy": "none",
            "attack_task_carried_routing_policy": "none",
        },
        legitimate_prompts=baseline,
        attack_prompts=baseline,
    )


def test_system_only_attack_request_has_no_task_carried_forward_note(
    monkeypatch,
) -> None:
    calls: list[dict] = []

    async def fake_episode_runner(**kwargs):
        calls.append(kwargs)
        is_benign = str(kwargs["episode_id"]).endswith("-BENIGN")
        audit_events = (
            (
                SimpleNamespace(
                    event_type="LOCAL_TOOL_EXECUTED",
                    metadata={"action": "get_file_by_id"},
                ),
            )
            if is_benign
            else ()
        )
        return SimpleNamespace(
            target_effect_verified=True,
            audit_events=audit_events,
            messages_sent=1,
            usage={"total_tokens": 10},
            as_dict=lambda: {"episode_id": kwargs["episode_id"]},
        )

    monkeypatch.setattr(
        "gatepath.mixed_workflow.run_request_response_episode_async",
        fake_episode_runner,
    )
    baseline = get_request_response_prompt_bundle(
        "en",
        version=f"{REQUEST_RESPONSE_REVISED_BASELINE_PROMPT_VERSION}-en",
    )
    result = asyncio.run(
        run_mixed_workflow_episode_async(
            episode_id="system-only-mixed-unit",
            condition="normal_plus_attack",
            regime="open_loop",
            normal_case=workflow_case_by_id("workspace-w10-i01"),
            attack_case=workflow_case_by_id("workspace-w10-i02"),
            graph=workflow_graph_by_id("workspace-single-route"),
            legitimate_prompts=baseline,
            attack_prompts=baseline,
            budget=RequestResponseBudget(
                max_messages=64,
                max_tool_iterations=4,
                max_runtime_seconds=180,
            ),
            response_visibility=ResponseVisibility.HIDDEN,
            handoff_continuation=HandoffContinuation.TERMINAL_AFTER_SEND,
            model="TEST",
            base_url="http://127.0.0.1:1/v1",
            temperature=0.0,
        )
    )
    assert len(calls) == 2
    assert "TEAM_WORKFLOW_NOTE" not in result.normal_request_content
    assert result.attack_request_content is not None
    assert "TEAM_WORKFLOW_NOTE" not in result.attack_request_content


def test_mixed_workflow_store_reports_benign_attack_and_joint_rates(
    tmp_path: Path,
) -> None:
    specs = _build_specs()
    selected = (specs[0], specs[3])
    store = MixedWorkflowBatchStore(
        run_root=tmp_path,
        exp_id="mixed-unit",
        config_payload=_load_config(),
        episode_specs=selected,
    )
    store.prepare(command="python mixed.py")
    store.write_completed(
        spec=selected[0],
        elapsed_seconds=1.0,
        report={
            "benign_workflow_success": True,
            "attack_success": False,
            "joint_success": False,
        },
    )
    store.write_completed(
        spec=selected[1],
        elapsed_seconds=2.0,
        report={
            "benign_workflow_success": True,
            "attack_success": True,
            "joint_success": True,
        },
    )
    summary = store.refresh_summary()
    open_loop = summary["by_condition_and_regime"][
        "normal_plus_attack__open_loop"
    ]
    natural = summary["by_condition_and_regime"][
        "normal_plus_attack__natural_feedback"
    ]
    assert open_loop["benign_success_rate"] == 1.0
    assert open_loop["attack_success_rate"] == 0.0
    assert open_loop["joint_success_rate"] == 0.0
    assert natural["benign_success_rate"] == 1.0
    assert natural["attack_success_rate"] == 1.0
    assert natural["joint_success_rate"] == 1.0


def test_normal_plus_attack_starts_both_requests_in_one_shared_environment(
    monkeypatch,
) -> None:
    started = 0
    both_started = asyncio.Event()
    calls: list[dict] = []

    async def fake_episode_runner(**kwargs):
        nonlocal started
        calls.append(kwargs)
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1.0)
        is_benign = str(kwargs["episode_id"]).endswith("-BENIGN")
        audit_events = (
            (
                SimpleNamespace(
                    event_type="LOCAL_TOOL_EXECUTED",
                    metadata={"action": "get_file_by_id"},
                ),
            )
            if is_benign
            else ()
        )
        return SimpleNamespace(
            target_effect_verified=True,
            audit_events=audit_events,
            messages_sent=1,
            usage={"total_tokens": 10},
            as_dict=lambda: {"episode_id": kwargs["episode_id"]},
        )

    monkeypatch.setattr(
        "gatepath.mixed_workflow.run_request_response_episode_async",
        fake_episode_runner,
    )
    result = asyncio.run(
        run_mixed_workflow_episode_async(
            episode_id="mixed-concurrency-unit",
            condition="normal_plus_attack",
            regime="natural_feedback",
            normal_case=workflow_case_by_id("workspace-w10-i01"),
            attack_case=workflow_case_by_id("workspace-w10-i02"),
            graph=workflow_graph_by_id("workspace-recoverable-branch"),
            legitimate_prompts=get_request_response_prompt_bundle(
                "en",
                version=(
                    f"{REQUEST_RESPONSE_REVISED_BASELINE_PROMPT_VERSION}-en"
                ),
            ),
            attack_prompts=get_request_response_prompt_bundle(
                "en",
                version=(
                    f"{REQUEST_RESPONSE_REVISED_FORWARD_PROMPT_VERSION}-en"
                ),
            ),
            budget=RequestResponseBudget(
                max_messages=64,
                max_tool_iterations=4,
                max_runtime_seconds=180,
            ),
            response_visibility=ResponseVisibility.NATURAL,
            handoff_continuation=HandoffContinuation.RETURN_TO_SENDER,
            model="TEST",
            base_url="http://127.0.0.1:1/v1",
            temperature=0.0,
        )
    )
    assert len(calls) == 2
    assert all(call["executor"] is calls[0]["executor"] for call in calls)
    assert all(call["reset_executor"] is False for call in calls)
    assert result.benign_workflow_success is True
    assert result.attack_success is True
    assert result.joint_success is True
    assert result.attack_request_content is not None
    assert "TEAM_WORKFLOW_NOTE" not in result.normal_request_content
    assert "TEAM_WORKFLOW_NOTE" in result.attack_request_content
