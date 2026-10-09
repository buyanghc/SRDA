"""请求—响应正式批量实验的计划、增量保存、恢复与汇总。

本模块不调用模型。它只负责冻结一次批次的精确 episode 清单，并将每个
episode 原子保存为独立 JSON。实际执行入口位于
``scripts/run_request_response_batch.py``。
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .request_response_protocol import (
    EntryAgentMode,
    HandoffContinuation,
    ResponseVisibility,
)
from .target_catalog import find_formal_target
from .target_instances import FormalTargetInstance


REQUEST_RESPONSE_BATCH_SCHEMA_VERSION = 1
_SAFE_EXP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class BatchSeedPair:
    """一次重复所使用的图随机种子与角色随机种子。"""

    repetition_id: int
    graph_seed: int
    role_seed: int

    def __post_init__(self) -> None:
        if self.repetition_id < 0:
            raise ValueError("repetition_id 不能为负数。")

    def as_dict(self) -> dict[str, int]:
        return {
            "repetition_id": self.repetition_id,
            "graph_seed": self.graph_seed,
            "role_seed": self.role_seed,
        }


@dataclass(frozen=True, slots=True)
class RequestResponseEpisodeSpec:
    """正式批次中一个不可再变化的 episode。"""

    instance_id: str
    suite_name: str
    graph_name: str
    repetition_id: int
    graph_seed: int
    role_seed: int

    @property
    def episode_key(self) -> str:
        return (
            f"{self.instance_id}__{_graph_slug(self.graph_name)}__"
            f"r{self.repetition_id:03d}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_key": self.episode_key,
            "instance_id": self.instance_id,
            "suite_name": self.suite_name,
            "graph_name": self.graph_name,
            "repetition_id": self.repetition_id,
            "graph_seed": self.graph_seed,
            "role_seed": self.role_seed,
        }


def build_batch_episode_specs(
    *,
    instances: Sequence[FormalTargetInstance],
    graph_names: Sequence[str],
    seed_pairs: Sequence[BatchSeedPair],
    max_episodes: int | None = None,
) -> tuple[RequestResponseEpisodeSpec, ...]:
    """按稳定顺序建立显式 episode 计划，可安全截取用于小规模运行。"""

    if not instances:
        raise ValueError("至少需要一个目标实例。")
    if not graph_names:
        raise ValueError("至少需要一张图。")
    if not seed_pairs:
        raise ValueError("至少需要一个 seed pair。")
    if len(set(graph_names)) != len(graph_names):
        raise ValueError("graph_names 不能重复。")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes 必须是正整数。")

    specs = [
        RequestResponseEpisodeSpec(
            instance_id=instance.instance_id,
            suite_name=instance.suite_name,
            graph_name=graph_name,
            repetition_id=seed_pair.repetition_id,
            graph_seed=seed_pair.graph_seed,
            role_seed=seed_pair.role_seed,
        )
        for instance in instances
        for graph_name in graph_names
        for seed_pair in seed_pairs
    ]
    if max_episodes is not None:
        specs = specs[:max_episodes]
    return tuple(specs)


@dataclass(frozen=True, slots=True)
class RequestResponseBatchConfig:
    """同一 exp_id 恢复运行时必须完全一致的配置。"""

    exp_id: str
    formal_config_id: str
    formal_config_sha256: str
    condition: str
    entry_agent_mode: str
    response_visibility: str
    model: str
    base_url: str
    temperature: float
    prompt_language: str
    prompt_version: str
    max_messages: int
    max_tool_iterations: int
    max_runtime_seconds: float
    target_manifest_sha256: str
    agentdojo_benchmark_version: str
    code_commit: str
    episode_specs: tuple[RequestResponseEpisodeSpec, ...]
    handoff_continuation: str | None = None
    target_selection_sha256: str | None = None
    api_key_env: str | None = None
    thinking_mode: str | None = None
    reasoning_effort: str | None = None
    minimum_cny_balance: float | None = None

    def __post_init__(self) -> None:
        if not _SAFE_EXP_ID.fullmatch(self.exp_id):
            raise ValueError(
                "exp_id 只能包含字母、数字、点、下划线和连字符，"
                "且必须以字母或数字开头。"
            )
        if not self.formal_config_id.strip() or not self.condition.strip():
            raise ValueError("formal_config_id 和 condition 不能为空。")
        if not self.model.strip() or not self.base_url.strip():
            raise ValueError("model 和 base_url 不能为空。")
        if self.api_key_env is not None and not self.api_key_env.strip():
            raise ValueError("api_key_env 不能为空字符串。")
        if self.thinking_mode not in {None, "disabled", "enabled"}:
            raise ValueError("thinking_mode 必须为 disabled 或 enabled。")
        if self.thinking_mode == "disabled" and self.reasoning_effort is not None:
            raise ValueError("关闭 thinking 时不能设置 reasoning_effort。")
        is_qwen = self.model.casefold().startswith("qwen")
        is_gptoss = self.model.casefold() in {"gpt-oss-20b", "openai/gpt-oss-20b"}
        if is_gptoss and (self.thinking_mode != "enabled" or self.reasoning_effort != "low"):
            raise ValueError("GPT-OSS campaign requires enabled thinking and low reasoning effort.")
        is_deepseek = self.base_url.rstrip("/").casefold() in {
            "https://api.deepseek.com",
            "https://api.deepseek.com/v1",
        }
        if self.thinking_mode == "enabled":
            if is_qwen and self.reasoning_effort is not None:
                raise ValueError("Qwen thinking 不使用 reasoning_effort。")
            if is_deepseek and self.reasoning_effort not in {
                "low",
                "high",
                "max",
            }:
                raise ValueError(
                    "启用 DeepSeek thinking 时 reasoning_effort 必须为 "
                    "low、high 或 max。"
                )
            if not is_qwen and not is_deepseek and not is_gptoss:
                raise ValueError("thinking_mode 当前仅支持 Qwen 和 DeepSeek。")
        if self.thinking_mode is None and self.reasoning_effort is not None:
            raise ValueError("设置 reasoning_effort 时必须显式启用 thinking。")
        if (
            self.minimum_cny_balance is not None
            and self.minimum_cny_balance < 0
        ):
            raise ValueError("minimum_cny_balance 不能为负数。")
        if self.minimum_cny_balance is not None and self.api_key_env is None:
            raise ValueError("余额保护要求显式设置 api_key_env。")
        EntryAgentMode(self.entry_agent_mode)
        ResponseVisibility(self.response_visibility)
        if self.handoff_continuation is not None:
            HandoffContinuation(self.handoff_continuation)
        if self.max_messages <= 0 or self.max_tool_iterations <= 0:
            raise ValueError("消息预算和工具轮数必须为正整数。")
        if self.max_runtime_seconds <= 0:
            raise ValueError("运行时间预算必须为正数。")
        if not self.episode_specs:
            raise ValueError("批量实验至少需要一个 episode。")
        keys = [spec.episode_key for spec in self.episode_specs]
        if len(keys) != len(set(keys)):
            raise ValueError("episode_specs 包含重复 episode_key。")

    @property
    def total_planned(self) -> int:
        return len(self.episode_specs)

    @property
    def instance_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(spec.instance_id for spec in self.episode_specs))

    @property
    def graph_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(spec.graph_name for spec in self.episode_specs))

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": REQUEST_RESPONSE_BATCH_SCHEMA_VERSION,
            "experiment_type": "AUTOGEN_NESTED_REQUEST_RESPONSE_BATCH",
            "condition": self.condition,
            "entry_agent_mode": self.entry_agent_mode,
            "response_visibility": self.response_visibility,
            "exp_id": self.exp_id,
            "formal_config_id": self.formal_config_id,
            "formal_config_sha256": self.formal_config_sha256,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "prompt_language": self.prompt_language,
            "prompt_version": self.prompt_version,
            "budget": {
                "max_messages": self.max_messages,
                "max_tool_iterations": self.max_tool_iterations,
                "max_runtime_seconds": self.max_runtime_seconds,
            },
            "target_manifest_sha256": self.target_manifest_sha256,
            "agentdojo_benchmark_version": self.agentdojo_benchmark_version,
            "code_commit": self.code_commit,
            "instance_ids": list(self.instance_ids),
            "graph_names": list(self.graph_names),
            "total_planned": self.total_planned,
            "episode_specs": [spec.as_dict() for spec in self.episode_specs],
        }
        if self.handoff_continuation is not None:
            payload["handoff_continuation"] = self.handoff_continuation
        if self.target_selection_sha256 is not None:
            payload["target_selection_sha256"] = (
                self.target_selection_sha256
            )
        if self.api_key_env is not None:
            payload["api_key_env"] = self.api_key_env
        if self.thinking_mode is not None:
            payload["thinking_mode"] = self.thinking_mode
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.minimum_cny_balance is not None:
            payload["minimum_cny_balance"] = self.minimum_cny_balance
        return payload


class RequestResponseBatchStore:
    """可靠保存批次状态；COMPLETED 跳过，ERROR 在恢复时重试。"""

    def __init__(
        self,
        run_root: str | Path,
        config: RequestResponseBatchConfig,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_root) / config.exp_id
        self.episodes_dir = self.run_dir / "episodes"
        self.metrics_dir = self.run_dir / "metrics"
        self.logs_dir = self.run_dir / "logs"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.config_path = self.run_dir / "config.json"
        self.plan_path = self.run_dir / "plan.json"
        self.command_path = self.run_dir / "command.sh"
        self.summary_path = self.metrics_dir / "summary.json"
        self.episodes_jsonl_path = self.metrics_dir / "episodes.jsonl"
        self.episodes_csv_path = self.metrics_dir / "episodes.csv"
        self.manifest_path = self.run_dir / "MANIFEST.md"
        self.checkpoint_ledger_path = (
            self.run_dir / "checkpoint-ledger.jsonl"
        )

    def prepare(self, *, command: str) -> None:
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        expected = self.config.as_dict()
        if self.config_path.exists():
            existing = _read_json(self.config_path)
            if existing != expected:
                raise RuntimeError(
                    "BATCH_CONFIG_MISMATCH：同一 exp_id 已存在不同配置。"
                )
        else:
            _atomic_write_json(self.config_path, expected)
        _atomic_write_json(
            self.plan_path,
            {
                "schema_version": REQUEST_RESPONSE_BATCH_SCHEMA_VERSION,
                "exp_id": self.config.exp_id,
                "total_planned": self.config.total_planned,
                "episodes": [
                    spec.as_dict() for spec in self.config.episode_specs
                ],
            },
        )
        _atomic_write_text(self.command_path, f"{command}\n")
        self.checkpoint_ledger_path.touch(exist_ok=True)
        self.refresh_artifacts(status="prepared")

    def episode_path(self, spec: RequestResponseEpisodeSpec) -> Path:
        return self.episodes_dir / f"{spec.episode_key}.json"

    def is_completed(self, spec: RequestResponseEpisodeSpec) -> bool:
        path = self.episode_path(spec)
        if not path.exists():
            return False
        try:
            record = _read_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            return False
        return record.get("status") == "COMPLETED"

    def write_completed(
        self,
        *,
        spec: RequestResponseEpisodeSpec,
        elapsed_seconds: float,
        report: Mapping[str, Any],
    ) -> None:
        _atomic_write_json(
            self.episode_path(spec),
            {
                "schema_version": REQUEST_RESPONSE_BATCH_SCHEMA_VERSION,
                **spec.as_dict(),
                "status": "COMPLETED",
                "elapsed_seconds": elapsed_seconds,
                "report": dict(report),
            },
        )

    def write_error(
        self,
        *,
        spec: RequestResponseEpisodeSpec,
        elapsed_seconds: float,
        error_type: str,
        error_message: str,
    ) -> None:
        _atomic_write_json(
            self.episode_path(spec),
            {
                "schema_version": REQUEST_RESPONSE_BATCH_SCHEMA_VERSION,
                **spec.as_dict(),
                "status": "ERROR",
                "elapsed_seconds": elapsed_seconds,
                "error": {
                    "type": error_type,
                    "message": error_message,
                },
            },
        )

    def refresh_artifacts(
        self,
        *,
        status: str | None = None,
    ) -> dict[str, Any]:
        records = self._ordered_existing_records()
        _atomic_write_text(
            self.episodes_jsonl_path,
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ),
        )
        _atomic_write_text(
            self.episodes_csv_path,
            _render_episode_csv(records),
        )
        summary = self._build_summary(records)
        if status is None:
            if summary["completed"] == summary["total_planned"]:
                status = "completed"
            elif summary["errors"] > 0:
                status = "needs_attention"
            else:
                status = "running"
        summary["status"] = status
        _atomic_write_json(self.summary_path, summary)
        _atomic_write_text(self.manifest_path, self._render_manifest(summary))
        return summary

    def _ordered_existing_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for spec in self.config.episode_specs:
            path = self.episode_path(spec)
            if not path.exists():
                continue
            try:
                records.append(_read_json(path))
            except (OSError, json.JSONDecodeError, ValueError):
                records.append(
                    {
                        "schema_version": REQUEST_RESPONSE_BATCH_SCHEMA_VERSION,
                        **spec.as_dict(),
                        "status": "ERROR",
                        "error": {
                            "type": "INVALID_EPISODE_FILE",
                            "message": f"无法读取 {path.name}",
                        },
                    }
                )
        return records

    def _build_summary(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        completed_records = [
            record for record in records
            if record.get("status") == "COMPLETED"
        ]
        reports = [
            record["report"]
            for record in completed_records
            if isinstance(record.get("report"), Mapping)
        ]
        errors = sum(record.get("status") == "ERROR" for record in records)
        elapsed = [
            float(record.get("elapsed_seconds", 0.0))
            for record in completed_records
        ]
        messages = [int(report.get("messages_sent", 0)) for report in reports]
        tokens = [
            int(report.get("usage", {}).get("total_tokens", 0))
            for report in reports
            if isinstance(report.get("usage"), Mapping)
        ]
        off_target_executions = [
            _off_target_sensitive_tool_executions(report)
            for report in reports
        ]
        off_target_episode_count = sum(
            bool(executions) for executions in off_target_executions
        )
        off_target_execution_count = sum(
            len(executions) for executions in off_target_executions
        )
        successes_with_off_target = sum(
            bool(report.get("attack_success")) and bool(executions)
            for report, executions in zip(
                reports,
                off_target_executions,
                strict=True,
            )
        )
        failures_with_off_target = sum(
            not bool(report.get("attack_success")) and bool(executions)
            for report, executions in zip(
                reports,
                off_target_executions,
                strict=True,
            )
        )
        termination_counts = Counter(
            str(report.get("termination_reason")) for report in reports
        )
        completed = len(completed_records)
        return {
            "schema_version": REQUEST_RESPONSE_BATCH_SCHEMA_VERSION,
            "exp_id": self.config.exp_id,
            "condition": self.config.condition,
            "total_planned": self.config.total_planned,
            "completed": completed,
            "errors": errors,
            "pending_without_record": (
                self.config.total_planned - completed - errors
            ),
            "unfinished_including_errors": (
                self.config.total_planned - completed
            ),
            "attack_successes": sum(
                bool(report.get("attack_success")) for report in reports
            ),
            "attack_success_rate_completed": _rate(
                sum(bool(report.get("attack_success")) for report in reports),
                completed,
            ),
            "target_agents_reached": sum(
                bool(report.get("target_agent_reached"))
                for report in reports
            ),
            "target_tools_dispatched": sum(
                bool(report.get("target_tool_dispatched"))
                for report in reports
            ),
            "target_tool_dispatch_count": sum(
                int(report.get("target_tool_dispatch_count", 0))
                for report in reports
            ),
            "sensitive_tool_dispatch_count": sum(
                int(report.get("sensitive_tool_dispatch_count", 0))
                for report in reports
            ),
            "repeated_sensitive_tool_dispatch_count": sum(
                int(
                    report.get(
                        "repeated_sensitive_tool_dispatch_count",
                        0,
                    )
                )
                for report in reports
            ),
            "target_effects_verified": sum(
                bool(report.get("target_effect_verified"))
                for report in reports
            ),
            "episodes_with_off_target_sensitive_tool_execution": (
                off_target_episode_count
            ),
            "off_target_sensitive_tool_execution_rate_completed": _rate(
                off_target_episode_count,
                completed,
            ),
            "off_target_sensitive_tool_executions": (
                off_target_execution_count
            ),
            "attack_successes_with_off_target_sensitive_tool_execution": (
                successes_with_off_target
            ),
            "target_failures_with_off_target_sensitive_tool_execution": (
                failures_with_off_target
            ),
            "infrastructure_errors": sum(
                bool(report.get("infrastructure_error"))
                for report in reports
            ),
            "message_budget_exhausted_episodes": sum(
                bool(report.get("message_budget_exhausted"))
                for report in reports
            ),
            "runtime_budget_exhausted_episodes": sum(
                bool(report.get("runtime_budget_exhausted"))
                for report in reports
            ),
            "episodes_with_cycle_closing_message": sum(
                int(report.get("cycle_closing_message_count", 0)) > 0
                for report in reports
            ),
            "cycle_closing_message_count": sum(
                int(report.get("cycle_closing_message_count", 0))
                for report in reports
            ),
            "repeated_node_visit_count": sum(
                int(report.get("repeated_node_visit_count", 0))
                for report in reports
            ),
            "repeated_edge_traversal_count": sum(
                int(report.get("repeated_edge_traversal_count", 0))
                for report in reports
            ),
            "termination_reason_counts": dict(
                sorted(termination_counts.items())
            ),
            "total_messages_sent": sum(messages),
            "mean_messages_sent": (
                None if not messages else sum(messages) / len(messages)
            ),
            "total_tokens": sum(tokens),
            "mean_total_tokens": (
                None if not tokens else sum(tokens) / len(tokens)
            ),
            "total_elapsed_seconds": sum(elapsed),
            "mean_elapsed_seconds": (
                None if not elapsed else sum(elapsed) / len(elapsed)
            ),
            "by_suite": self._group_summary(records, field="suite_name"),
            "by_graph": self._group_summary(records, field="graph_name"),
        }

    def _group_summary(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        field: str,
    ) -> dict[str, Any]:
        planned = Counter(
            str(getattr(spec, field)) for spec in self.config.episode_specs
        )
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record.get(field))].append(record)
        result: dict[str, Any] = {}
        for name in planned:
            completed = [
                record for record in groups.get(name, [])
                if record.get("status") == "COMPLETED"
            ]
            reports = [
                record["report"]
                for record in completed
                if isinstance(record.get("report"), Mapping)
            ]
            successes = sum(
                bool(report.get("attack_success")) for report in reports
            )
            off_target_executions = [
                _off_target_sensitive_tool_executions(report)
                for report in reports
            ]
            off_target_episode_count = sum(
                bool(executions) for executions in off_target_executions
            )
            successes_with_off_target = sum(
                bool(report.get("attack_success")) and bool(executions)
                for report, executions in zip(
                    reports,
                    off_target_executions,
                    strict=True,
                )
            )
            failures_with_off_target = sum(
                not bool(report.get("attack_success"))
                and bool(executions)
                for report, executions in zip(
                    reports,
                    off_target_executions,
                    strict=True,
                )
            )
            result[name] = {
                "planned": planned[name],
                "completed": len(completed),
                "errors": sum(
                    record.get("status") == "ERROR"
                    for record in groups.get(name, [])
                ),
                "attack_successes": successes,
                "attack_success_rate_completed": _rate(
                    successes,
                    len(completed),
                ),
                "message_budget_exhausted_episodes": sum(
                    bool(report.get("message_budget_exhausted"))
                    for report in reports
                ),
                "runtime_budget_exhausted_episodes": sum(
                    bool(report.get("runtime_budget_exhausted"))
                    for report in reports
                ),
                "episodes_with_cycle_closing_message": sum(
                    int(report.get("cycle_closing_message_count", 0)) > 0
                    for report in reports
                ),
                "cycle_closing_message_count": sum(
                    int(report.get("cycle_closing_message_count", 0))
                    for report in reports
                ),
                "repeated_node_visit_count": sum(
                    int(report.get("repeated_node_visit_count", 0))
                    for report in reports
                ),
                "repeated_edge_traversal_count": sum(
                    int(report.get("repeated_edge_traversal_count", 0))
                    for report in reports
                ),
                "target_tool_dispatch_count": sum(
                    int(report.get("target_tool_dispatch_count", 0))
                    for report in reports
                ),
                "sensitive_tool_dispatch_count": sum(
                    int(report.get("sensitive_tool_dispatch_count", 0))
                    for report in reports
                ),
                "repeated_sensitive_tool_dispatch_count": sum(
                    int(
                        report.get(
                            "repeated_sensitive_tool_dispatch_count",
                            0,
                        )
                    )
                    for report in reports
                ),
                "episodes_with_off_target_sensitive_tool_execution": (
                    off_target_episode_count
                ),
                "off_target_sensitive_tool_execution_rate_completed": _rate(
                    off_target_episode_count,
                    len(completed),
                ),
                "off_target_sensitive_tool_executions": sum(
                    len(executions)
                    for executions in off_target_executions
                ),
                (
                    "attack_successes_with_off_target_"
                    "sensitive_tool_execution"
                ): successes_with_off_target,
                (
                    "target_failures_with_off_target_"
                    "sensitive_tool_execution"
                ): failures_with_off_target,
            }
        return result

    def _render_manifest(self, summary: Mapping[str, Any]) -> str:
        return (
            f"# Run Manifest: {self.config.exp_id}\n\n"
            f"- Objective: GatePath synchronous nested request-response batch\n"
            f"- Status: {summary['status']}\n"
            f"- Formal config: {self.config.formal_config_id}\n"
            f"- Model: {self.config.model}\n"
            f"- AgentDojo: {self.config.agentdojo_benchmark_version}\n"
            f"- Code commit: {self.config.code_commit}\n"
            f"- Planned episodes: {summary['total_planned']}\n"
            f"- Completed episodes: {summary['completed']}\n"
            f"- Errors: {summary['errors']}\n"
            f"- Attack successes: {summary['attack_successes']}\n"
            "- Message-budget exhausted episodes: "
            f"{summary['message_budget_exhausted_episodes']}\n"
            "- Runtime-budget exhausted episodes: "
            f"{summary['runtime_budget_exhausted_episodes']}\n"
            "- Cycle-closing messages: "
            f"{summary['cycle_closing_message_count']}\n"
            "- Episodes with off-target sensitive tool execution: "
            f"{summary['episodes_with_off_target_sensitive_tool_execution']}\n"
            "- Off-target sensitive tool executions: "
            f"{summary['off_target_sensitive_tool_executions']}\n"
            "- Attack successes with off-target sensitive execution: "
            f"{summary['attack_successes_with_off_target_sensitive_tool_execution']}\n"
            "- Target failures with off-target sensitive execution: "
            f"{summary['target_failures_with_off_target_sensitive_tool_execution']}\n"
            f"- Configuration: config.json\n"
            f"- Explicit plan: plan.json\n"
            f"- Per-episode records: episodes/\n"
            f"- Summary: metrics/summary.json\n"
            f"- CSV summary: metrics/episodes.csv\n"
            f"- Machine-readable ledger: metrics/episodes.jsonl\n"
            f"- Checkpoints: none expected for inference-only runs\n"
        )


def _render_episode_csv(records: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "episode_key",
        "instance_id",
        "suite_name",
        "graph_name",
        "repetition_id",
        "graph_seed",
        "role_seed",
        "status",
        "attack_success",
        "target_agent_reached",
        "target_tool_dispatched",
        "target_tool_dispatch_count",
        "target_effect_verified",
        "sensitive_tool_dispatch_count",
        "repeated_sensitive_tool_dispatch_count",
        "off_target_sensitive_tool_executed",
        "off_target_sensitive_tool_execution_count",
        "off_target_sensitive_actions",
        "infrastructure_error",
        "termination_reason",
        "message_budget_exhausted",
        "runtime_budget_exhausted",
        "repeated_node_visit_count",
        "repeated_edge_traversal_count",
        "cycle_closing_message_count",
        "messages_sent",
        "total_tokens",
        "elapsed_seconds",
        "error_type",
        "error_message",
    )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for record in records:
        report = (
            record.get("report")
            if isinstance(record.get("report"), Mapping)
            else {}
        )
        error = (
            record.get("error")
            if isinstance(record.get("error"), Mapping)
            else {}
        )
        usage = (
            report.get("usage")
            if isinstance(report.get("usage"), Mapping)
            else {}
        )
        off_target_executions = (
            _off_target_sensitive_tool_executions(report)
        )
        writer.writerow(
            {
                "episode_key": record.get("episode_key"),
                "instance_id": record.get("instance_id"),
                "suite_name": record.get("suite_name"),
                "graph_name": record.get("graph_name"),
                "repetition_id": record.get("repetition_id"),
                "graph_seed": record.get("graph_seed"),
                "role_seed": record.get("role_seed"),
                "status": record.get("status"),
                "attack_success": report.get("attack_success"),
                "target_agent_reached": report.get("target_agent_reached"),
                "target_tool_dispatched": report.get(
                    "target_tool_dispatched"
                ),
                "target_tool_dispatch_count": report.get(
                    "target_tool_dispatch_count"
                ),
                "target_effect_verified": report.get(
                    "target_effect_verified"
                ),
                "sensitive_tool_dispatch_count": report.get(
                    "sensitive_tool_dispatch_count"
                ),
                "repeated_sensitive_tool_dispatch_count": report.get(
                    "repeated_sensitive_tool_dispatch_count"
                ),
                "off_target_sensitive_tool_executed": bool(
                    off_target_executions
                ),
                "off_target_sensitive_tool_execution_count": len(
                    off_target_executions
                ),
                "off_target_sensitive_actions": json.dumps(
                    [
                        execution.get("executed_action")
                        for execution in off_target_executions
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "infrastructure_error": report.get(
                    "infrastructure_error"
                ),
                "termination_reason": report.get("termination_reason"),
                "message_budget_exhausted": report.get(
                    "message_budget_exhausted"
                ),
                "runtime_budget_exhausted": report.get(
                    "runtime_budget_exhausted"
                ),
                "repeated_node_visit_count": report.get(
                    "repeated_node_visit_count"
                ),
                "repeated_edge_traversal_count": report.get(
                    "repeated_edge_traversal_count"
                ),
                "cycle_closing_message_count": report.get(
                    "cycle_closing_message_count"
                ),
                "messages_sent": report.get("messages_sent"),
                "total_tokens": usage.get("total_tokens"),
                "elapsed_seconds": record.get("elapsed_seconds"),
                "error_type": error.get("type"),
                "error_message": error.get("message"),
            }
        )
    return output.getvalue()


def _off_target_sensitive_tool_executions(
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """读取新字段，或从0.1版audit_events向后兼容地推导。"""

    if "off_target_sensitive_tool_executions" in report:
        stored = report.get("off_target_sensitive_tool_executions")
        if not isinstance(stored, Sequence) or isinstance(
            stored,
            (str, bytes),
        ):
            return ()
        return tuple(
            dict(execution)
            for execution in stored
            if isinstance(execution, Mapping)
        )

    target = report.get("target")
    if not isinstance(target, Mapping):
        return ()
    suite_name = str(target.get("suite_name", ""))
    target_action = str(target.get("action", ""))
    events = report.get("audit_events")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return ()

    executions: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if event.get("event_type") != "LOCAL_TOOL_EXECUTED":
            continue
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        action = str(metadata.get("action", ""))
        if action == target_action:
            continue
        sensitive_type = find_formal_target(suite_name, action)
        if sensitive_type is None:
            continue
        result = metadata.get("result")
        if not isinstance(result, Mapping):
            continue
        if (
            not bool(result.get("tool_dispatched"))
            or result.get("error") is not None
            or bool(result.get("infrastructure_error"))
        ):
            continue
        arguments = metadata.get("arguments")
        executions.append(
            {
                "request_id": event.get("request_id"),
                "agent_id": event.get("receiver_id"),
                "target_action": target_action,
                "executed_action": action,
                "sensitive_target_type_id": (
                    sensitive_type.target_type_id
                ),
                "arguments": (
                    dict(arguments)
                    if isinstance(arguments, Mapping)
                    else {}
                ),
                "result": dict(result),
                "evidence_basis": (
                    "TOOL_DISPATCHED_WITHOUT_REPORTED_ERROR"
                ),
                "state_change_verification": (
                    "NOT_INDEPENDENTLY_VERIFIED"
                ),
            }
        )
    return tuple(executions)


def _graph_slug(graph_name: str) -> str:
    normalized = graph_name.strip().lower().replace(" ", "")
    if normalized == "gdag":
        return "gdag"
    if normalized in {
        "g32-x2",
        "g32-p2",
        "g32-x3",
        "g32-p3",
        "g33-edge",
        "g33-cycle",
        "g33-x2",
        "g33-p2",
        "g33-x3",
        "g33-p3",
        "g16-acyclic-control",
        "g16-cycle",
    }:
        return normalized
    match = re.fullmatch(r"g\((\d+),(\d+)\)", normalized)
    if match is None:
        raise ValueError(f"无法生成图文件名：{graph_name!r}")
    return f"g-b{match.group(1)}-d{match.group(2)}"


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} 顶层必须是 JSON 对象。")
    return raw


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
