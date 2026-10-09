"""Preview or run paired normal-workflow-under-attack episodes."""

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

from gatepath.mixed_workflow import run_mixed_workflow_episode
from gatepath.mixed_workflow_batch import (
    MixedWorkflowBatchStore,
    build_mixed_workflow_episode_specs,
)
from gatepath.request_response_protocol import (
    HandoffContinuation,
    RequestResponseBudget,
    RequestResponsePromptBundle,
    ResponseVisibility,
    get_request_response_prompt_bundle,
)
from gatepath.workflow_graphs import workflow_graph_by_id
from gatepath.workflow_scenarios import workflow_case_by_id


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "semantic_workflow_mixed_full.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or execute paired Normal-only and Normal+attack semantic "
            "workflow episodes. Execution requires --execute."
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
        pairings = _mapping(domain_config, "attack_pair_by_normal_pair")
        selections.append(
            (
                tuple(str(item) for item in domain_config["graph_ids"]),
                tuple(str(item) for item in domain_config["normal_case_pair_ids"]),
                {str(key): str(value) for key, value in pairings.items()},
            )
        )
    regimes = _mapping(frozen, "regimes")
    specs = build_mixed_workflow_episode_specs(
        domain_selections=selections,
        conditions=tuple(str(item) for item in experiment["conditions"]),
        regimes=tuple(str(item) for item in regimes),
        repetitions=int(experiment["repetitions"]),
        max_episodes=args.max_episodes,
    )
    store = MixedWorkflowBatchStore(
        run_root=args.run_root,
        exp_id=args.exp_id,
        config_payload=frozen,
        episode_specs=specs,
    )
    store.prepare(command=shlex.join([sys.executable, *sys.argv]))
    print(
        json.dumps(
            {
                "event": "MIXED_WORKFLOW_PLAN_PREPARED",
                "exp_id": args.exp_id,
                "execute": args.execute,
                "total_planned": len(specs),
                "first_episodes": [spec.as_dict() for spec in specs[:6]],
                "run_dir": str(store.run_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not args.execute:
        return

    model_config = _mapping(frozen, "model")
    prompt_versions = _mapping(model_config, "prompt_versions")
    legitimate_prompts = get_request_response_prompt_bundle(
        str(model_config["prompt_language"]),
        version=str(prompt_versions["legitimate"]),
    )
    attack_prompts = get_request_response_prompt_bundle(
        str(model_config["prompt_language"]),
        version=str(prompt_versions["attack"]),
    )
    _validate_prompt_policies(
        experiment=experiment,
        legitimate_prompts=legitimate_prompts,
        attack_prompts=attack_prompts,
    )

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
            normal_case = workflow_case_by_id(spec.normal_pair_id)
            attack_case = workflow_case_by_id(spec.attack_pair_id)
            graph = workflow_graph_by_id(spec.graph_id)
            if not (
                graph.suite_name
                == normal_case.suite_name
                == attack_case.suite_name
            ):
                raise RuntimeError("Mixed graph and case suites do not match.")
            regime = _mapping(regimes, spec.regime)
            report = run_mixed_workflow_episode(
                episode_id=f"{args.exp_id}-{spec.episode_key}",
                condition=spec.condition,
                regime=spec.regime,
                normal_case=normal_case,
                attack_case=attack_case,
                graph=graph,
                legitimate_prompts=legitimate_prompts,
                attack_prompts=attack_prompts,
                budget=RequestResponseBudget(
                    max_messages=int(common_budget["max_messages_per_request"]),
                    max_tool_iterations=int(regime["max_tool_iterations"]),
                    max_runtime_seconds=float(
                        common_budget["max_runtime_seconds_per_request"]
                    ),
                ),
                response_visibility=ResponseVisibility(
                    str(regime["response_visibility"])
                ),
                handoff_continuation=HandoffContinuation(
                    str(regime["handoff_continuation"])
                ),
                model=str(model_config["name"]),
                base_url=base_url,
                temperature=float(model_config["temperature"]),
            )
            payload = report.as_dict()
            store.write_completed(
                spec=spec,
                elapsed_seconds=time.monotonic() - started,
                report=payload,
            )
            print(
                json.dumps(
                    {
                        "event": "MIXED_WORKFLOW_EPISODE_COMPLETED",
                        "episode_key": spec.episode_key,
                        "benign_workflow_success": (
                            payload["benign_workflow_success"]
                        ),
                        "attack_success": payload["attack_success"],
                        "joint_success": payload["joint_success"],
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
                "event": "MIXED_WORKFLOW_BATCH_FINISHED",
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
        raise SystemExit("Mixed workflow config must be a frozen mapping.")
    if raw.get("schema_version") != "1.0":
        raise SystemExit("Unsupported mixed workflow config version.")
    return raw


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise SystemExit(f"Config section {key!r} must be a mapping.")
    return value


def _validate_prompt_policies(
    *,
    experiment: Mapping[str, Any],
    legitimate_prompts: RequestResponsePromptBundle,
    attack_prompts: RequestResponsePromptBundle,
) -> None:
    legitimate_policy = str(
        experiment.get("legitimate_task_carried_routing_policy", "")
    )
    attack_policy = str(
        experiment.get("attack_task_carried_routing_policy", "")
    )
    if legitimate_policy != "none" or legitimate_prompts.workflow_note is not None:
        raise RuntimeError("Normal workflow must use system-level handoff only.")
    if attack_policy == "none":
        if attack_prompts.workflow_note is not None:
            raise RuntimeError(
                "System-only attack must not carry a task-level routing note."
            )
    elif attack_policy == "exact_revised_forward_instruction":
        if attack_prompts.workflow_note is None:
            raise RuntimeError(
                "Forward attack requires the task-carried routing note."
            )
    else:
        raise RuntimeError(
            f"Unsupported attack task-carried routing policy: {attack_policy!r}."
        )
    if (
        legitimate_prompts.normal_agent_system_prompt
        != attack_prompts.normal_agent_system_prompt
    ):
        raise RuntimeError("Both requests must share the victim system policy.")
    if not (
        legitimate_prompts.preserve_canonical_work_item
        and attack_prompts.preserve_canonical_work_item
    ):
        raise RuntimeError("Both requests require canonical payload preservation.")


if __name__ == "__main__":
    main()
