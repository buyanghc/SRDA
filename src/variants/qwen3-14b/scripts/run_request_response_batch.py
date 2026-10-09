"""可筛选、可预览、可恢复的GatePath请求—响应正式批量入口。

示例：

仅预览Workspace、两张图、一个seed的计划（不会调用模型）：

    .venv/bin/python scripts/run_request_response_batch.py \
        --exp-id workspace-two-graphs-preview \
        --suite workspace \
        --graph 'G(1,2)' \
        --graph 'G(2,2)'

确认后执行相同计划：

    .venv/bin/python scripts/run_request_response_batch.py \
        --exp-id workspace-two-graphs-preview \
        --suite workspace \
        --graph 'G(1,2)' \
        --graph 'G(2,2)' \
        --execute

完整3600个episode必须显式写出all、all-graphs、5次重复和execute：

    .venv/bin/python scripts/run_request_response_batch.py \
        --exp-id request-response-formal-v1 \
        --suite all \
        --all-graphs \
        --repetitions 5 \
        --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gatepath.agentdojo_adapter import DEFAULT_AGENTDOJO_BENCHMARK_VERSION
from gatepath.deepseek_balance import (
    DeepSeekBalanceError,
    fetch_deepseek_balance,
)
from gatepath.experiment_graphs import build_named_topology
from gatepath.request_response_batch import (
    BatchSeedPair,
    RequestResponseBatchConfig,
    RequestResponseBatchStore,
    RequestResponseEpisodeSpec,
    build_batch_episode_specs,
)
from gatepath.request_response_protocol import (
    EntryAgentMode,
    HandoffContinuation,
    RequestResponseBudget,
    ResponseVisibility,
    get_request_response_prompt_bundle,
)
from gatepath.request_response_runner import run_request_response_episode
from gatepath.target_instances import (
    load_formal_target_instances,
)


DEFAULT_FORMAL_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "request_response_formal_v2.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按显式选择运行GatePath请求—响应批次；默认只预览，"
            "只有加入--execute才会调用模型。"
        )
    )
    parser.add_argument("--exp-id", required=True)
    parser.add_argument(
        "--formal-config",
        type=Path,
        default=DEFAULT_FORMAL_CONFIG_PATH,
    )
    parser.add_argument("--run-root", type=Path, default=Path("runs"))

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--suite",
        choices=("all", "workspace", "banking"),
        help="选择全部目标、Workspace或Banking。",
    )
    target_group.add_argument(
        "--instance-id",
        action="append",
        dest="instance_ids",
        help="只选择指定目标实例；可以重复使用。",
    )
    parser.add_argument(
        "--instance-limit",
        type=int,
        help="在选定目标中只取前N个。",
    )

    graph_group = parser.add_mutually_exclusive_group(required=True)
    graph_group.add_argument(
        "--all-graphs",
        action="store_true",
        help="选择冻结配置中的全部图。",
    )
    graph_group.add_argument(
        "--graph",
        action="append",
        dest="graphs",
        help="只选择指定图；可以重复使用，例如--graph 'G(1,2)'。",
    )
    parser.add_argument(
        "--graph-limit",
        type=int,
        help="在选定图中只取前N张。",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="使用冻结seed pair中的前N组；默认1，正式全量为5。",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="最终计划只取前N个episode，主要用于冒烟测试和分块运行。",
    )
    parser.add_argument(
        "--episode-key-file",
        type=Path,
        help=(
            "只运行文件中逐行列出的episode_key；用于不规则子集的"
            "定向恢复或敏感性实验。"
        ),
    )
    parser.add_argument(
        "--new-episode-limit",
        type=int,
        help=(
            "本次进程最多尝试N个尚未完成的episode，但保持完整计划不变；"
            "用于可恢复、条件交替的正式运行。"
        ),
    )
    parser.add_argument(
        "--base-url",
        help="覆盖OpenAI兼容模型服务地址；记录到批次配置。",
    )
    parser.add_argument(
        "--model",
        help=(
            "覆盖冻结模型名称，用于保持其他设置不变的跨模型复现；"
            "覆盖值会记录到批次配置。"
        ),
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        help=(
            "显式覆盖冻结配置中的单episode运行时间预算；"
            "覆盖值会记录到批次配置。"
        ),
    )
    parser.add_argument(
        "--api-key-env",
        help=(
            "从指定环境变量读取API密钥；只记录变量名，绝不记录密钥值。"
        ),
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("disabled", "enabled"),
        help=(
            "显式记录并控制DeepSeek thinking模式；其他provider暂不支持。"
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "high", "max"),
        help="DeepSeek thinking启用时的推理强度。",
    )
    parser.add_argument(
        "--minimum-cny-balance",
        type=float,
        help=(
            "每个新episode开始前检查DeepSeek余额；若总余额低于该值，"
            "不写episode错误并以可恢复暂停状态退出。"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正调用模型；不加时只创建并显示计划。",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="单个episode异常后继续；默认保存错误并停止。",
    )
    parser.add_argument(
        "--summary-every",
        type=int,
        default=10,
        help=(
            "每完成N个新episode重建一次全量汇总；"
            "每个episode文件仍会立即保存。默认10。"
        ),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="允许脏工作树执行，仅供开发验证；正式实验禁止使用。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_positive_optional("--instance-limit", args.instance_limit)
    _validate_positive_optional("--graph-limit", args.graph_limit)
    _validate_positive_optional("--max-episodes", args.max_episodes)
    _validate_positive_optional(
        "--max-runtime-seconds",
        args.max_runtime_seconds,
    )
    _validate_positive_optional(
        "--new-episode-limit",
        args.new_episode_limit,
    )
    if args.summary_every <= 0:
        raise SystemExit("--summary-every 必须是正整数。")
    if args.repetitions <= 0:
        raise SystemExit("--repetitions 必须是正整数。")
    if args.api_key_env is not None and not args.api_key_env.strip():
        raise SystemExit("--api-key-env 不能为空字符串。")
    if args.thinking_mode == "enabled" and args.reasoning_effort is None:
        raise SystemExit(
            "--thinking-mode enabled 必须同时显式设置 --reasoning-effort。"
        )
    if args.thinking_mode != "enabled" and args.reasoning_effort is not None:
        raise SystemExit(
            "--reasoning-effort 只能与 --thinking-mode enabled 同时使用。"
        )
    if args.minimum_cny_balance is not None:
        if args.minimum_cny_balance < 0:
            raise SystemExit("--minimum-cny-balance 不能为负数。")
        if args.api_key_env is None:
            raise SystemExit(
                "--minimum-cny-balance 必须与 --api-key-env 同时使用。"
            )

    api_key = "EMPTY"
    if args.execute and args.api_key_env is not None:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key.strip():
            raise SystemExit(
                f"API_KEY_ENV_MISSING：环境变量 {args.api_key_env!r} "
                "不存在或为空。"
            )

    frozen = _load_frozen_config(args.formal_config)
    targets = _mapping(frozen, "targets")
    target_registry = _resolve_project_path(str(targets["registry"]))
    all_instances = load_formal_target_instances(target_registry)
    expected_registry_instance_count = int(
        targets.get(
            "expected_registry_instance_count",
            targets["expected_instance_count"],
        )
    )
    if len(all_instances) != expected_registry_instance_count:
        raise SystemExit(
            "TARGET_INSTANCE_COUNT_MISMATCH：冻结配置要求 "
            f"{expected_registry_instance_count} 个注册实例，实际读取 "
            f"{len(all_instances)} 个。"
        )
    frozen_instances, target_selection_sha256 = (
        _select_frozen_instance_pool(
            all_instances,
            targets=targets,
        )
    )
    selected_instances = _select_instances(
        frozen_instances,
        suite=args.suite,
        instance_ids=args.instance_ids,
        limit=args.instance_limit,
    )
    selected_graphs = _select_graphs(
        frozen,
        all_graphs=args.all_graphs,
        requested=args.graphs,
        limit=args.graph_limit,
    )
    seed_pairs = _select_seed_pairs(frozen, args.repetitions)
    specs = build_batch_episode_specs(
        instances=selected_instances,
        graph_names=selected_graphs,
        seed_pairs=seed_pairs,
        max_episodes=args.max_episodes,
    )
    if args.episode_key_file is not None:
        specs = _select_episode_specs_from_file(
            specs,
            args.episode_key_file,
        )

    base_url = args.base_url
    if base_url is None:
        port = os.environ.get("GATEPATH_QWEN_PORT", "8010")
        base_url = f"http://127.0.0.1:{port}/v1"

    model = _mapping(frozen, "model")
    model_name = args.model or str(model["name"])
    budget = _mapping(frozen, "budget")
    experiment = _mapping(frozen, "experiment")
    config = RequestResponseBatchConfig(
        exp_id=args.exp_id,
        formal_config_id=str(frozen["config_id"]),
        formal_config_sha256=_sha256(args.formal_config),
        condition=str(experiment["condition"]),
        entry_agent_mode=str(
            experiment.get("entry_agent_mode", "deterministic")
        ),
        response_visibility=str(
            experiment.get("response_visibility", "natural")
        ),
        model=model_name,
        base_url=base_url,
        temperature=float(model["temperature"]),
        prompt_language=str(model["prompt_language"]),
        prompt_version=str(model["prompt_version"]),
        max_messages=int(budget["max_messages"]),
        max_tool_iterations=int(budget["max_tool_iterations"]),
        max_runtime_seconds=(
            float(args.max_runtime_seconds)
            if args.max_runtime_seconds is not None
            else float(budget["max_runtime_seconds"])
        ),
        target_manifest_sha256=_sha256(target_registry),
        agentdojo_benchmark_version=DEFAULT_AGENTDOJO_BENCHMARK_VERSION,
        code_commit=_git_commit(),
        episode_specs=specs,
        handoff_continuation=(
            None
            if experiment.get("handoff_continuation") is None
            else str(experiment["handoff_continuation"])
        ),
        target_selection_sha256=target_selection_sha256,
        api_key_env=args.api_key_env,
        thinking_mode=args.thinking_mode,
        reasoning_effort=args.reasoning_effort,
        minimum_cny_balance=args.minimum_cny_balance,
    )
    store = RequestResponseBatchStore(args.run_root, config)
    store.prepare(command=shlex.join([sys.executable, *sys.argv]))
    preview = {
        "event": "BATCH_PLAN_PREPARED",
        "exp_id": config.exp_id,
        "run_dir": str(store.run_dir),
        "execute": args.execute,
        "condition": config.condition,
        "entry_agent_mode": config.entry_agent_mode,
        "response_visibility": config.response_visibility,
        "model": config.model,
        "api_key_env": config.api_key_env,
        "thinking_mode": config.thinking_mode,
        "reasoning_effort": config.reasoning_effort,
        "minimum_cny_balance": config.minimum_cny_balance,
        "handoff_continuation": (
            config.handoff_continuation
            or HandoffContinuation.RETURN_TO_SENDER.value
        ),
        "target_registry": str(target_registry),
        "episode_key_file": (
            None
            if args.episode_key_file is None
            else str(args.episode_key_file)
        ),
        "total_planned": config.total_planned,
        "instance_count": len(config.instance_ids),
        "suites": list(dict.fromkeys(spec.suite_name for spec in specs)),
        "graphs": list(config.graph_names),
        "repetition_ids": list(
            dict.fromkeys(spec.repetition_id for spec in specs)
        ),
        "first_episodes": [
            spec.as_dict() for spec in config.episode_specs[:5]
        ],
    }
    print(json.dumps(preview, ensure_ascii=False), flush=True)
    if not args.execute:
        store.refresh_artifacts(status="prepared")
        return

    if _git_is_dirty() and not args.allow_dirty:
        store.refresh_artifacts(status="blocked_dirty_worktree")
        raise SystemExit(
            "正式执行要求Git工作树干净。请先提交代码；"
            "开发验证可显式使用--allow-dirty。"
        )

    instances_by_id = {
        instance.instance_id: instance for instance in all_instances
    }
    prompts = get_request_response_prompt_bundle(
        config.prompt_language,
        version=config.prompt_version,
    )
    if prompts.version != config.prompt_version:
        raise RuntimeError(
            "PROMPT_VERSION_MISMATCH：冻结配置与代码提示词版本不一致。"
        )
    episode_budget = RequestResponseBudget(
        max_messages=config.max_messages,
        max_tool_iterations=config.max_tool_iterations,
        max_runtime_seconds=config.max_runtime_seconds,
    )
    initial_summary = store.refresh_artifacts(status="running")
    completed_count = int(initial_summary["completed"])
    completed_since_summary = 0
    attempted_new = 0

    for spec in config.episode_specs:
        if store.is_completed(spec):
            print(
                json.dumps(
                    {
                        "event": "EPISODE_SKIPPED_COMPLETED",
                        "episode_key": spec.episode_key,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        if (
            args.new_episode_limit is not None
            and attempted_new >= args.new_episode_limit
        ):
            break
        attempted_new += 1

        if config.minimum_cny_balance is not None:
            try:
                balance = fetch_deepseek_balance(
                    api_key=api_key,
                    base_url=config.base_url,
                )
            except DeepSeekBalanceError as exc:
                summary = store.refresh_artifacts(
                    status="paused_balance_check_failed"
                )
                print(
                    json.dumps(
                        {
                            "event": "BATCH_PAUSED_BALANCE_CHECK_FAILED",
                            "reason": str(exc),
                            "completed": summary["completed"],
                            "errors": summary["errors"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                raise SystemExit(75) from exc
            if (
                not balance.is_available
                or balance.total_cny
                < Decimal(str(config.minimum_cny_balance))
            ):
                summary = store.refresh_artifacts(
                    status="paused_insufficient_balance"
                )
                print(
                    json.dumps(
                        {
                            "event": "BATCH_PAUSED_INSUFFICIENT_BALANCE",
                            "balance": balance.as_dict(),
                            "minimum_cny_balance": (
                                config.minimum_cny_balance
                            ),
                            "completed": summary["completed"],
                            "errors": summary["errors"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                raise SystemExit(75)

        instance = instances_by_id[spec.instance_id]
        topology = build_named_topology(
            spec.graph_name,
            seed=spec.graph_seed,
        )
        episode_id = f"{config.exp_id}-{spec.episode_key.lower()}"
        started = time.monotonic()
        try:
            report = run_request_response_episode(
                instance=instance,
                topology=topology,
                prompts=prompts,
                condition=config.condition,
                budget=episode_budget,
                episode_id=episode_id,
                role_seed=spec.role_seed,
                model=config.model,
                base_url=config.base_url,
                api_key=api_key,
                temperature=config.temperature,
                thinking_mode=config.thinking_mode,
                reasoning_effort=config.reasoning_effort,
                entry_agent_mode=config.entry_agent_mode,
                response_visibility=config.response_visibility,
                handoff_continuation=(
                    config.handoff_continuation
                    or HandoffContinuation.RETURN_TO_SENDER.value
                ),
            )
        except KeyboardInterrupt:
            store.refresh_artifacts(status="interrupted")
            raise
        except Exception as exc:
            if (
                config.minimum_cny_balance is not None
                and _is_insufficient_balance_error(exc)
            ):
                summary = store.refresh_artifacts(
                    status="paused_insufficient_balance"
                )
                print(
                    json.dumps(
                        {
                            "event": "BATCH_PAUSED_INSUFFICIENT_BALANCE",
                            "reason": "provider_rejected_for_balance",
                            "completed": summary["completed"],
                            "errors": summary["errors"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                raise SystemExit(75) from exc
            elapsed = time.monotonic() - started
            store.write_error(
                spec=spec,
                elapsed_seconds=elapsed,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            summary = store.refresh_artifacts(status="needs_attention")
            print(
                json.dumps(
                    {
                        "event": "EPISODE_ERROR",
                        "episode_key": spec.episode_key,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "completed": summary["completed"],
                        "errors": summary["errors"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not args.continue_on_error:
                raise
            continue

        elapsed = time.monotonic() - started
        payload = report.as_dict()
        store.write_completed(
            spec=spec,
            elapsed_seconds=elapsed,
            report=payload,
        )
        completed_count += 1
        completed_since_summary += 1
        if (
            completed_since_summary >= args.summary_every
            or completed_count == config.total_planned
        ):
            summary = store.refresh_artifacts()
            completed_count = int(summary["completed"])
            completed_since_summary = 0
        print(
            json.dumps(
                {
                    "event": "EPISODE_COMPLETED",
                    "episode_key": spec.episode_key,
                    "attack_success": report.attack_success,
                    "messages_sent": report.messages_sent,
                    "elapsed_seconds": round(elapsed, 3),
                    "completed": completed_count,
                    "total_planned": config.total_planned,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if report.infrastructure_error and not args.continue_on_error:
            store.refresh_artifacts(status="needs_attention")
            raise RuntimeError(
                "REQUEST_RESPONSE_INFRASTRUCTURE_ERROR："
                "当前episode报告基础设施错误。"
            )

    final_summary = store.refresh_artifacts()
    final_event = (
        "BATCH_FINISHED"
        if final_summary["completed"] == final_summary["total_planned"]
        else "BATCH_CHUNK_FINISHED"
    )
    print(
        json.dumps(
            {
                "event": final_event,
                **final_summary,
                "attempted_new_this_process": attempted_new,
                "summary_path": str(store.summary_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if final_summary["errors"] > 0:
        raise SystemExit(1)


def _is_insufficient_balance_error(exc: BaseException) -> bool:
    """Recognize provider balance exhaustion without recording attack ERROR."""

    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "insufficient balance",
        "insufficient_balance",
        "balance is insufficient",
        "余额不足",
        "402 payment required",
        "error code: 402",
        "status_code=402",
    )
    return any(marker in text for marker in markers)


def _load_frozen_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("正式配置顶层必须是映射。")
    if raw.get("status") != "frozen":
        raise SystemExit("正式配置必须处于frozen状态。")
    model = _mapping(raw, "model")
    if model.get("enable_thinking") is not False:
        raise SystemExit("第一阶段正式配置必须固定enable_thinking=false。")
    if model.get("parallel_tool_calls") is not True:
        raise SystemExit("正式配置与当前运行时parallel_tool_calls不一致。")
    experiment = _mapping(raw, "experiment")
    try:
        EntryAgentMode(
            str(experiment.get("entry_agent_mode", "deterministic"))
        )
        ResponseVisibility(
            str(experiment.get("response_visibility", "natural"))
        )
        if experiment.get("handoff_continuation") is not None:
            HandoffContinuation(str(experiment["handoff_continuation"]))
    except ValueError as exc:
        raise SystemExit(f"无效的入口或回复可见性配置：{exc}") from exc
    return raw


def _select_frozen_instance_pool(
    all_instances: Sequence[Any],
    *,
    targets: Mapping[str, Any],
) -> tuple[tuple[Any, ...], str | None]:
    selection = str(targets.get("selection", "all_registry_instances"))
    expected_selected_count = int(targets["expected_instance_count"])
    if selection == "all_registry_instances":
        if len(all_instances) != expected_selected_count:
            raise SystemExit(
                "FROZEN_TARGET_SELECTION_COUNT_MISMATCH："
                f"配置要求 {expected_selected_count} 个目标，"
                f"注册表包含 {len(all_instances)} 个。"
            )
        return tuple(all_instances), None

    if selection != "fixed_selection_manifest":
        raise SystemExit(f"未知冻结目标选择策略：{selection}")

    selection_path = _resolve_project_path(
        str(targets["selection_manifest"])
    )
    expected_sha256 = str(targets["selection_manifest_sha256"])
    actual_sha256 = _sha256(selection_path)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "TARGET_SELECTION_SHA256_MISMATCH："
            f"期望 {expected_sha256}，实际 {actual_sha256}。"
        )
    raw = yaml.safe_load(selection_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("status") != "frozen":
        raise SystemExit("目标选择清单必须是frozen映射。")
    raw_ids = raw.get("instance_ids")
    if not isinstance(raw_ids, list) or not all(
        isinstance(item, str) and item for item in raw_ids
    ):
        raise SystemExit("目标选择清单缺少有效instance_ids。")
    if len(raw_ids) != len(set(raw_ids)):
        raise SystemExit("目标选择清单包含重复instance_id。")
    by_id = {instance.instance_id: instance for instance in all_instances}
    unknown = [item for item in raw_ids if item not in by_id]
    if unknown:
        raise SystemExit(
            "目标选择清单包含未知实例：" + ", ".join(unknown)
        )
    selected = tuple(by_id[item] for item in raw_ids)
    if len(selected) != expected_selected_count:
        raise SystemExit(
            "FROZEN_TARGET_SELECTION_COUNT_MISMATCH："
            f"配置要求 {expected_selected_count} 个目标，"
            f"选择清单包含 {len(selected)} 个。"
        )
    expected_per_type = raw.get("expected_instances_per_target_type")
    if expected_per_type is not None:
        counts: dict[str, int] = {}
        for instance in selected:
            counts[instance.target_type_id] = (
                counts.get(instance.target_type_id, 0) + 1
            )
        if set(counts.values()) != {int(expected_per_type)}:
            raise SystemExit(
                "目标选择清单没有为每种能力提供相同数量的实例。"
            )
    return selected, actual_sha256


def _select_instances(
    all_instances: Sequence[Any],
    *,
    suite: str | None,
    instance_ids: Sequence[str] | None,
    limit: int | None,
) -> tuple[Any, ...]:
    by_id = {instance.instance_id: instance for instance in all_instances}
    if instance_ids:
        if len(instance_ids) != len(set(instance_ids)):
            raise SystemExit("--instance-id 不能重复。")
        unknown = [item for item in instance_ids if item not in by_id]
        if unknown:
            raise SystemExit(f"未知正式目标实例：{', '.join(unknown)}")
        selected = [by_id[item] for item in instance_ids]
    else:
        if suite is None:
            raise SystemExit("必须选择--suite或--instance-id。")
        selected = [
            instance
            for instance in all_instances
            if suite == "all" or instance.suite_name == suite
        ]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise SystemExit("没有选中任何目标实例。")
    return tuple(selected)


def _select_episode_specs_from_file(
    all_specs: Sequence[RequestResponseEpisodeSpec],
    selection_path: Path,
) -> tuple[RequestResponseEpisodeSpec, ...]:
    """Select an exact, ordered episode subset from a newline manifest."""

    if not selection_path.is_absolute():
        selection_path = PROJECT_ROOT / selection_path
    try:
        raw_lines = selection_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(
            f"无法读取--episode-key-file：{selection_path}: {exc}"
        ) from exc
    requested = [
        line.strip()
        for line in raw_lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not requested:
        raise SystemExit("--episode-key-file没有包含任何episode_key。")
    if len(requested) != len(set(requested)):
        raise SystemExit("--episode-key-file包含重复episode_key。")

    by_key = {spec.episode_key: spec for spec in all_specs}
    unknown = [key for key in requested if key not in by_key]
    if unknown:
        preview = ", ".join(unknown[:10])
        suffix = "" if len(unknown) <= 10 else f" 等{len(unknown)}项"
        raise SystemExit(
            "--episode-key-file包含不在当前冻结计划中的key："
            f"{preview}{suffix}"
        )
    return tuple(by_key[key] for key in requested)


def _select_graphs(
    frozen: Mapping[str, Any],
    *,
    all_graphs: bool,
    requested: Sequence[str] | None,
    limit: int | None,
) -> tuple[str, ...]:
    graph_config = _mapping(frozen, "graphs")
    allowed = tuple(str(item) for item in graph_config["names"])
    if all_graphs:
        selected = list(allowed)
    else:
        selected = list(requested or ())
        if len(selected) != len(set(selected)):
            raise SystemExit("--graph 不能重复。")
        unknown = [item for item in selected if item not in allowed]
        if unknown:
            raise SystemExit(
                "图不在冻结配置中："
                f"{', '.join(unknown)}；允许值：{', '.join(allowed)}"
            )
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise SystemExit("没有选中任何图。")
    return tuple(selected)


def _select_seed_pairs(
    frozen: Mapping[str, Any],
    repetitions: int,
) -> tuple[BatchSeedPair, ...]:
    randomization = _mapping(frozen, "randomization")
    raw_pairs = randomization.get("seed_pairs")
    if not isinstance(raw_pairs, list):
        raise SystemExit("冻结配置缺少seed_pairs。")
    if repetitions > len(raw_pairs):
        raise SystemExit(
            f"--repetitions 最大为{len(raw_pairs)}，当前为{repetitions}。"
        )
    selected: list[BatchSeedPair] = []
    for item in raw_pairs[:repetitions]:
        if not isinstance(item, Mapping):
            raise SystemExit("冻结配置中的每个seed pair必须是映射。")
        selected.append(
            BatchSeedPair(
                repetition_id=int(item["repetition_id"]),
                graph_seed=int(item["graph_seed"]),
                role_seed=int(item["role_seed"]),
            )
        )
    return tuple(selected)


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise SystemExit(f"正式配置缺少映射：{key}")
    return value


def _validate_positive_optional(
    name: str,
    value: int | float | None,
) -> None:
    if value is not None and value <= 0:
        raise SystemExit(f"{name} 必须是正数。")


def _resolve_project_path(value: str) -> Path:
    """将冻结配置中的项目相对路径解析为绝对路径。"""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _git_is_dirty() -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(output.strip())


if __name__ == "__main__":
    main()
