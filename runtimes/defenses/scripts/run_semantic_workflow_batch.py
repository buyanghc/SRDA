"""Preview or run the Workspace/Banking semantic workflow experiment."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gatepath.request_response_protocol import (
    EntryAgentMode,
    HandoffContinuation,
    RequestResponseBudget,
    ResponseVisibility,
    get_request_response_prompt_bundle,
)
from gatepath.request_response_runner import run_request_response_episode
from gatepath.workflow_batch import (
    WorkflowBatchStore,
    build_workflow_episode_specs,
)
from gatepath.workflow_graphs import workflow_graph_by_id
from gatepath.workflow_scenarios import (
    build_workflow_request_content,
    prerequisite_actions_executed,
    workflow_case_by_id,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "semantic_workflow_realism.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or execute the fixed Workspace/Banking semantic workflow "
            "experiment. Execution requires --execute."
        )
    )
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--suite", choices=("all", "workspace", "banking"), default="all")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--new-episode-limit", type=int)
    parser.add_argument("--base-url")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name, value in (
        ("--max-episodes", args.max_episodes),
        ("--new-episode-limit", args.new_episode_limit),
    ):
        if value is not None and value <= 0:
            raise SystemExit(f"{name} must be positive.")

    frozen = _load_config(args.config)
    experiment = _mapping(frozen, "experiment")
    domains = [str(item) for item in experiment["domains"]]
    if args.suite != "all":
        domains = [args.suite]
    selections = []
    for domain in domains:
        domain_config = _mapping(frozen, domain)
        selections.append(
            (
                tuple(str(item) for item in domain_config["graph_ids"]),
                tuple(str(item) for item in domain_config["case_pair_ids"]),
            )
        )
    regimes = _mapping(frozen, "regimes")
    specs = build_workflow_episode_specs(
        domain_selections=selections,
        task_kinds=tuple(str(item) for item in experiment["task_kinds"]),
        regimes=tuple(str(item) for item in regimes),
        repetitions=int(experiment["repetitions"]),
        max_episodes=args.max_episodes,
    )
    store = WorkflowBatchStore(
        run_root=args.run_root,
        exp_id=args.exp_id,
        config_payload=frozen,
        episode_specs=specs,
    )
    store.prepare(command=shlex.join([sys.executable, *sys.argv]))
    preview = {
        "event": "SEMANTIC_WORKFLOW_PLAN_PREPARED",
        "exp_id": args.exp_id,
        "execute": args.execute,
        "total_planned": len(specs),
        "first_episodes": [spec.as_dict() for spec in specs[:6]],
        "run_dir": str(store.run_dir),
    }
    print(json.dumps(preview, ensure_ascii=False), flush=True)
    if not args.execute:
        return

    model_config = _mapping(frozen, "model")
    prompt_versions = _mapping(model_config, "prompt_versions")
    prompts_by_task_kind = {
        task_kind: get_request_response_prompt_bundle(
            str(model_config["prompt_language"]),
            version=str(prompt_versions[task_kind]),
        )
        for task_kind in ("legitimate", "attack")
    }
    legitimate_prompts = prompts_by_task_kind["legitimate"]
    attack_prompts = prompts_by_task_kind["attack"]
    if legitimate_prompts.workflow_note is not None:
        raise RuntimeError("Legitimate workflow tasks must use system handoff only.")
    if attack_prompts.workflow_note is None:
        raise RuntimeError("Attack workflow tasks require task-carried Forward.")
    if (
        legitimate_prompts.normal_agent_system_prompt
        != attack_prompts.normal_agent_system_prompt
    ):
        raise RuntimeError("Both realism groups must share the same system policy.")
    if not all(
        prompts.preserve_canonical_work_item
        for prompts in prompts_by_task_kind.values()
    ):
        raise RuntimeError("Semantic realism requires runtime task preservation.")

    base_url = args.base_url
    if base_url is None:
        port = os.environ.get("GATEPATH_QWEN_PORT", "8010")
        base_url = f"http://127.0.0.1:{port}/v1"
    common_budget = _mapping(frozen, "budget")
    attempted_new = 0

    for spec in specs:
        if store.is_completed(spec):
            continue
        if (
            args.new_episode_limit is not None
            and attempted_new >= args.new_episode_limit
        ):
            break
        attempted_new += 1
        started = time.monotonic()
        try:
            case = workflow_case_by_id(spec.pair_id)
            graph = workflow_graph_by_id(spec.graph_id)
            if graph.suite_name != case.suite_name:
                raise RuntimeError("Workflow graph and case suites do not match.")
            topology = graph.topology(task_kind=spec.task_kind)
            assigned_team = graph.assigned_team(topology)
            instance = case.target_instance()
            prompts = prompts_by_task_kind[spec.task_kind]
            request_content = build_workflow_request_content(
                case=case,
                task_kind=spec.task_kind,
                workflow_note=prompts.workflow_note,
            )
            regime = _mapping(regimes, spec.regime)
            entry_mode = (
                EntryAgentMode.LLM
                if spec.task_kind == "legitimate"
                else EntryAgentMode.DETERMINISTIC
            )
            report = run_request_response_episode(
                instance=instance,
                topology=topology,
                prompts=prompts,
                condition=(
                    f"SEMANTIC_{spec.task_kind.upper()}_"
                    f"{spec.regime.upper()}"
                ),
                budget=RequestResponseBudget(
                    max_messages=int(common_budget["max_messages"]),
                    max_tool_iterations=int(regime["max_tool_iterations"]),
                    max_runtime_seconds=float(
                        common_budget["max_runtime_seconds"]
                    ),
                ),
                episode_id=f"{args.exp_id}-{spec.episode_key}",
                model=str(model_config["name"]),
                base_url=base_url,
                temperature=float(model_config["temperature"]),
                entry_agent_mode=entry_mode,
                response_visibility=ResponseVisibility(
                    str(regime["response_visibility"])
                ),
                handoff_continuation=HandoffContinuation(
                    str(
                        regime.get(
                            "handoff_continuation",
                            HandoffContinuation.RETURN_TO_SENDER.value,
                        )
                    )
                ),
                assigned_team_override=assigned_team,
                request_content_override=request_content,
            )
            prerequisite_status = prerequisite_actions_executed(
                required_actions=case.legitimate_prerequisite_actions,
                audit_events=report.audit_events,
            )
            legitimate_success = (
                spec.task_kind == "legitimate"
                and report.target_effect_verified
                and all(prerequisite_status.values())
            )
            payload = {
                "workflow_schema_version": "1.0",
                "pair": case.as_dict(),
                "workflow_graph": graph.as_dict(),
                "task_kind": spec.task_kind,
                "regime": spec.regime,
                "entry_agent_mode": entry_mode.value,
                "request_content": request_content,
                "prerequisite_actions_executed": dict(prerequisite_status),
                "legitimate_workflow_success": legitimate_success,
                "attack_success": (
                    report.target_effect_verified
                    if spec.task_kind == "attack"
                    else None
                ),
                "request_response_report": report.as_dict(),
            }
            store.write_completed(
                spec=spec,
                elapsed_seconds=time.monotonic() - started,
                report=payload,
            )
            print(
                json.dumps(
                    {
                        "event": "SEMANTIC_WORKFLOW_EPISODE_COMPLETED",
                        "episode_key": spec.episode_key,
                        "legitimate_workflow_success": legitimate_success,
                        "attack_success": payload["attack_success"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            store.write_error(
                spec=spec,
                elapsed_seconds=time.monotonic() - started,
                error=exc,
            )
            store.refresh_summary(status="needs_attention")
            if not args.continue_on_error:
                raise

    summary = store.refresh_summary()
    print(
        json.dumps(
            {
                "event": "SEMANTIC_WORKFLOW_BATCH_FINISHED",
                **summary,
                "attempted_new_this_process": attempted_new,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("status") != "frozen":
        raise SystemExit("Semantic workflow config must be a frozen mapping.")
    if raw.get("schema_version") != "1.0":
        raise SystemExit("Unsupported semantic workflow config version.")
    return raw


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise SystemExit(f"Config section {key!r} must be a mapping.")
    return value


if __name__ == "__main__":
    main()
