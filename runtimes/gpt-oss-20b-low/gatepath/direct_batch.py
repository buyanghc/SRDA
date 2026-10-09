"""Direct-only 批量实验的配置、增量保存和汇总。

本模块不调用 Qwen，也不决定攻击策略。它只负责把每个 episode 的结果可靠地
保存到独立文件，并根据已有文件判断断点续跑位置。实际模型调用由
``scripts/run_direct_batch.py`` 负责。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DIRECT_BATCH_SCHEMA_VERSION = 2
_SAFE_EXP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class DirectBatchConfig:
    """一次 Direct-only 批量实验中禁止悄悄变化的配置。"""

    exp_id: str
    feedback_level: str
    model: str
    base_url: str
    temperature: float
    max_tokens: int
    max_parse_retries: int
    max_decisions: int
    repetitions: int
    instance_ids: tuple[str, ...]
    target_manifest_sha256: str
    agentdojo_benchmark_version: str
    prompt_version: str
    code_commit: str

    def __post_init__(self) -> None:
        if not _SAFE_EXP_ID.fullmatch(self.exp_id):
            raise ValueError(
                "exp_id 只能包含字母、数字、点、下划线和连字符，"
                "且必须以字母或数字开头。"
            )
        if self.feedback_level not in {"F0", "F1", "F2"}:
            raise ValueError("feedback_level 只能是 F0、F1 或 F2。")
        if not self.model.strip():
            raise ValueError("model 不能为空。")
        if not self.base_url.strip():
            raise ValueError("base_url 不能为空。")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须是正整数。")
        if self.max_parse_retries < 0:
            raise ValueError("max_parse_retries 不能为负数。")
        if self.max_decisions <= 0:
            raise ValueError("max_decisions 必须是正整数。")
        if self.repetitions <= 0:
            raise ValueError("repetitions 必须是正整数。")
        if not self.instance_ids:
            raise ValueError("批量实验至少需要一个目标实例。")
        if len(set(self.instance_ids)) != len(self.instance_ids):
            raise ValueError("instance_ids 不能包含重复项。")

    @property
    def total_planned(self) -> int:
        return len(self.instance_ids) * self.repetitions

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DIRECT_BATCH_SCHEMA_VERSION,
            "experiment_type": "DIRECT_ONLY",
            "exp_id": self.exp_id,
            "feedback_level": self.feedback_level,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_parse_retries": self.max_parse_retries,
            "max_decisions": self.max_decisions,
            "repetitions": self.repetitions,
            "instance_ids": list(self.instance_ids),
            "target_manifest_sha256": self.target_manifest_sha256,
            "agentdojo_benchmark_version": (
                self.agentdojo_benchmark_version
            ),
            "prompt_version": self.prompt_version,
            "code_commit": self.code_commit,
            "total_planned": self.total_planned,
        }


class DirectBatchStore:
    """一个可恢复的批量实验目录。

    每个 episode 使用独立 JSON 文件。文件先写入同目录临时文件，再原子替换，
    避免程序中断时留下半个 JSON。
    """

    def __init__(self, run_root: str | Path, config: DirectBatchConfig) -> None:
        self.config = config
        self.run_dir = Path(run_root) / config.exp_id
        self.episodes_dir = self.run_dir / "episodes"
        self.metrics_dir = self.run_dir / "metrics"
        self.logs_dir = self.run_dir / "logs"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.config_path = self.run_dir / "config.json"
        self.command_path = self.run_dir / "command.sh"
        self.summary_path = self.metrics_dir / "summary.json"
        self.episodes_jsonl_path = self.metrics_dir / "episodes.jsonl"
        self.manifest_path = self.run_dir / "MANIFEST.md"
        self.checkpoint_ledger_path = (
            self.run_dir / "checkpoint-ledger.jsonl"
        )

    def prepare(self, *, command: str) -> None:
        """创建目录，并拒绝用不同配置污染同一个 exp_id。"""

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

        _atomic_write_text(self.command_path, f"{command}\n")
        self.checkpoint_ledger_path.touch(exist_ok=True)
        self.refresh_artifacts(status="running")

    def episode_key(self, instance_id: str, repetition: int) -> str:
        if repetition <= 0:
            raise ValueError("repetition 从1开始。")
        return f"{instance_id}__r{repetition:03d}"

    def episode_path(self, instance_id: str, repetition: int) -> Path:
        return self.episodes_dir / (
            f"{self.episode_key(instance_id, repetition)}.json"
        )

    def is_completed(self, instance_id: str, repetition: int) -> bool:
        path = self.episode_path(instance_id, repetition)
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
        instance_id: str,
        repetition: int,
        elapsed_seconds: float,
        report: Mapping[str, Any],
    ) -> None:
        """保存一个正常返回的 episode；不等于攻击成功。"""

        record = {
            "schema_version": DIRECT_BATCH_SCHEMA_VERSION,
            "episode_key": self.episode_key(instance_id, repetition),
            "instance_id": instance_id,
            "repetition": repetition,
            "status": "COMPLETED",
            "elapsed_seconds": elapsed_seconds,
            "report": dict(report),
        }
        _atomic_write_json(
            self.episode_path(instance_id, repetition),
            record,
        )

    def write_error(
        self,
        *,
        instance_id: str,
        repetition: int,
        elapsed_seconds: float,
        error_type: str,
        error_message: str,
    ) -> None:
        """保存异常；ERROR 在恢复运行时会再次尝试，而不是被跳过。"""

        record = {
            "schema_version": DIRECT_BATCH_SCHEMA_VERSION,
            "episode_key": self.episode_key(instance_id, repetition),
            "instance_id": instance_id,
            "repetition": repetition,
            "status": "ERROR",
            "elapsed_seconds": elapsed_seconds,
            "error": {
                "type": error_type,
                "message": error_message,
            },
        }
        _atomic_write_json(
            self.episode_path(instance_id, repetition),
            record,
        )

    def refresh_artifacts(self, *, status: str | None = None) -> dict[str, Any]:
        """从 episode 文件重新生成 JSONL、汇总和 MANIFEST。"""

        records = self._ordered_existing_records()
        _atomic_write_text(
            self.episodes_jsonl_path,
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                + "\n"
                for record in records
            ),
        )
        summary = self._build_summary(records)
        if status is None:
            if summary["completed"] == summary["total_planned"]:
                status = "completed"
            elif summary["safety_invariant_violations"] > 0:
                status = "safety_violation"
            elif summary["errors"] > 0:
                status = "needs_attention"
            else:
                status = "running"
        summary["status"] = status
        _atomic_write_json(self.summary_path, summary)
        _atomic_write_text(
            self.manifest_path,
            self._render_manifest(summary),
        )
        return summary

    def _ordered_existing_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for instance_id in self.config.instance_ids:
            for repetition in range(1, self.config.repetitions + 1):
                path = self.episode_path(instance_id, repetition)
                if not path.exists():
                    continue
                try:
                    record = _read_json(path)
                except (OSError, json.JSONDecodeError, ValueError):
                    record = {
                        "schema_version": DIRECT_BATCH_SCHEMA_VERSION,
                        "episode_key": self.episode_key(
                            instance_id,
                            repetition,
                        ),
                        "instance_id": instance_id,
                        "repetition": repetition,
                        "status": "ERROR",
                        "error": {
                            "type": "INVALID_EPISODE_FILE",
                            "message": f"无法读取 {path.name}",
                        },
                    }
                records.append(record)
        return records

    def _build_summary(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        completed_records = [
            record
            for record in records
            if record.get("status") == "COMPLETED"
        ]
        errors = sum(record.get("status") == "ERROR" for record in records)
        reports = [
            record["report"]
            for record in completed_records
            if isinstance(record.get("report"), Mapping)
        ]
        termination_counts = Counter(
            str(report.get("termination_reason"))
            for report in reports
        )
        safety_violations = sum(
            bool(report.get("attack_success"))
            or bool(report.get("target_tool_dispatched"))
            or bool(report.get("target_effect_verified"))
            for report in reports
        )
        elapsed = [
            float(record.get("elapsed_seconds", 0.0))
            for record in completed_records
        ]
        decisions = [
            int(report.get("budget", {}).get("decisions_used", 0))
            for report in reports
            if isinstance(report.get("budget"), Mapping)
        ]

        completed = len(completed_records)
        return {
            "schema_version": DIRECT_BATCH_SCHEMA_VERSION,
            "exp_id": self.config.exp_id,
            "condition": f"DIRECT_ONLY_{self.config.feedback_level}",
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
            "immediate_stops": sum(
                bool(report.get("immediate_stop")) for report in reports
            ),
            "target_tool_dispatched": sum(
                bool(report.get("target_tool_dispatched"))
                for report in reports
            ),
            "target_effect_verified": sum(
                bool(report.get("target_effect_verified"))
                for report in reports
            ),
            "infrastructure_errors": sum(
                bool(report.get("infrastructure_error"))
                for report in reports
            ),
            "safety_invariant_violations": safety_violations,
            "termination_reason_counts": dict(
                sorted(termination_counts.items())
            ),
            "total_elapsed_seconds": sum(elapsed),
            "mean_elapsed_seconds": (
                None if not elapsed else sum(elapsed) / len(elapsed)
            ),
            "mean_decisions_used": (
                None if not decisions else sum(decisions) / len(decisions)
            ),
        }

    def _render_manifest(self, summary: Mapping[str, Any]) -> str:
        return (
            f"# Run Manifest: {self.config.exp_id}\n\n"
            f"- Objective: Direct-only "
            f"{self.config.feedback_level} batch validation\n"
            f"- Status: {summary['status']}\n"
            f"- Model: {self.config.model}\n"
            f"- AgentDojo: {self.config.agentdojo_benchmark_version}\n"
            f"- Code commit: {self.config.code_commit}\n"
            f"- Planned episodes: {summary['total_planned']}\n"
            f"- Completed episodes: {summary['completed']}\n"
            f"- Errors: {summary['errors']}\n"
            f"- Safety invariant violations: "
            f"{summary['safety_invariant_violations']}\n"
            f"- Configuration: config.json\n"
            f"- Per-episode records: episodes/\n"
            f"- Summary: metrics/summary.json\n"
            f"- Machine-readable ledger: metrics/episodes.jsonl\n"
            f"- Checkpoints: none expected for inference-only runs\n"
        )


def direct_report_has_safety_violation(report: Mapping[str, Any]) -> bool:
    """Direct-only 中三项任意为真都需要立即停止检查。"""

    return any(
        bool(report.get(field))
        for field in (
            "attack_success",
            "target_tool_dispatched",
            "target_effect_verified",
        )
    )


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
