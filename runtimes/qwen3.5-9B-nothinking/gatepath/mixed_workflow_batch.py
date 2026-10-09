"""Resumable storage for paired normal-workflow-under-attack experiments."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MIXED_WORKFLOW_BATCH_SCHEMA_VERSION = 1
MIXED_WORKFLOW_CONDITIONS = ("normal_only", "normal_plus_attack")
_SAFE_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class MixedWorkflowEpisodeSpec:
    normal_pair_id: str
    attack_pair_id: str
    graph_id: str
    condition: str
    regime: str
    repetition_id: int

    def __post_init__(self) -> None:
        for value in (
            self.normal_pair_id,
            self.attack_pair_id,
            self.graph_id,
            self.condition,
            self.regime,
        ):
            if not _SAFE_PART.fullmatch(value):
                raise ValueError(f"Unsafe mixed-workflow identifier: {value!r}.")
        if self.condition not in MIXED_WORKFLOW_CONDITIONS:
            raise ValueError(
                "condition must be normal_only or normal_plus_attack."
            )
        if self.normal_pair_id == self.attack_pair_id:
            raise ValueError("Normal and attack target instances must differ.")
        if self.repetition_id < 0:
            raise ValueError("repetition_id must not be negative.")

    @property
    def episode_key(self) -> str:
        return (
            f"{self.normal_pair_id}__attack-{self.attack_pair_id}__"
            f"{self.graph_id}__{self.condition}__{self.regime}__"
            f"r{self.repetition_id:03d}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_key": self.episode_key,
            "normal_pair_id": self.normal_pair_id,
            "attack_pair_id": self.attack_pair_id,
            "graph_id": self.graph_id,
            "condition": self.condition,
            "regime": self.regime,
            "repetition_id": self.repetition_id,
        }


def build_mixed_workflow_episode_specs(
    *,
    domain_selections: Sequence[
        tuple[Sequence[str], Sequence[str], Mapping[str, str]]
    ],
    conditions: Sequence[str],
    regimes: Sequence[str],
    repetitions: int,
    max_episodes: int | None = None,
) -> tuple[MixedWorkflowEpisodeSpec, ...]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    specs = tuple(
        MixedWorkflowEpisodeSpec(
            normal_pair_id=normal_pair_id,
            attack_pair_id=str(attack_pairing[normal_pair_id]),
            graph_id=graph_id,
            condition=condition,
            regime=regime,
            repetition_id=repetition_id,
        )
        for condition in conditions
        for graph_ids, normal_pair_ids, attack_pairing in domain_selections
        for graph_id in graph_ids
        for normal_pair_id in normal_pair_ids
        for regime in regimes
        for repetition_id in range(repetitions)
    )
    if not specs:
        raise ValueError("Mixed-workflow experiment plan must not be empty.")
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive.")
        return specs[:max_episodes]
    return specs


class MixedWorkflowBatchStore:
    def __init__(
        self,
        *,
        run_root: str | Path,
        exp_id: str,
        config_payload: Mapping[str, Any],
        episode_specs: Sequence[MixedWorkflowEpisodeSpec],
    ) -> None:
        if not _SAFE_PART.fullmatch(exp_id):
            raise ValueError("Unsafe mixed-workflow exp_id.")
        self.exp_id = exp_id
        self.config_payload = dict(config_payload)
        self.episode_specs = tuple(episode_specs)
        self.run_dir = Path(run_root) / exp_id
        self.episodes_dir = self.run_dir / "episodes"
        self.metrics_dir = self.run_dir / "metrics"
        self.logs_dir = self.run_dir / "logs"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.config_path = self.run_dir / "config.json"
        self.plan_path = self.run_dir / "plan.json"
        self.summary_path = self.metrics_dir / "summary.json"
        self.command_path = self.run_dir / "command.sh"
        self.manifest_path = self.run_dir / "MANIFEST.md"
        self.checkpoint_ledger_path = self.run_dir / "checkpoint-ledger.jsonl"

    def prepare(self, *, command: str = "") -> None:
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        expected = {
            "schema_version": MIXED_WORKFLOW_BATCH_SCHEMA_VERSION,
            "experiment_type": "PAIRED_MIXED_SEMANTIC_WORKFLOW_BATCH",
            "exp_id": self.exp_id,
            "formal_config": self.config_payload,
        }
        if self.config_path.exists():
            existing = json.loads(self.config_path.read_text(encoding="utf-8"))
            if existing != expected:
                raise RuntimeError(
                    "MIXED_WORKFLOW_BATCH_CONFIG_MISMATCH: exp_id already exists."
                )
        else:
            _atomic_write_json(self.config_path, expected)
        _atomic_write_json(
            self.plan_path,
            {
                "schema_version": MIXED_WORKFLOW_BATCH_SCHEMA_VERSION,
                "total_planned": len(self.episode_specs),
                "episodes": [spec.as_dict() for spec in self.episode_specs],
            },
        )
        _atomic_write_text(self.command_path, f"{command}\n")
        self.checkpoint_ledger_path.touch(exist_ok=True)
        self.refresh_summary(status="prepared")

    def episode_path(self, spec: MixedWorkflowEpisodeSpec) -> Path:
        return self.episodes_dir / f"{spec.episode_key}.json"

    def is_completed(self, spec: MixedWorkflowEpisodeSpec) -> bool:
        path = self.episode_path(spec)
        if not path.exists():
            return False
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("status") == "COMPLETED"
        except (OSError, json.JSONDecodeError):
            return False

    def write_completed(
        self,
        *,
        spec: MixedWorkflowEpisodeSpec,
        elapsed_seconds: float,
        report: Mapping[str, Any],
    ) -> None:
        _atomic_write_json(
            self.episode_path(spec),
            {
                "schema_version": MIXED_WORKFLOW_BATCH_SCHEMA_VERSION,
                **spec.as_dict(),
                "status": "COMPLETED",
                "elapsed_seconds": elapsed_seconds,
                "report": dict(report),
            },
        )

    def write_error(
        self,
        *,
        spec: MixedWorkflowEpisodeSpec,
        elapsed_seconds: float,
        error: Exception,
    ) -> None:
        _atomic_write_json(
            self.episode_path(spec),
            {
                "schema_version": MIXED_WORKFLOW_BATCH_SCHEMA_VERSION,
                **spec.as_dict(),
                "status": "ERROR",
                "elapsed_seconds": elapsed_seconds,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            },
        )

    def refresh_summary(self, *, status: str | None = None) -> dict[str, Any]:
        records = []
        for spec in self.episode_specs:
            path = self.episode_path(spec)
            if not path.exists():
                continue
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                records.append({**spec.as_dict(), "status": "ERROR"})

        completed = [record for record in records if record.get("status") == "COMPLETED"]
        errors = sum(record.get("status") == "ERROR" for record in records)
        grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {
                "completed": 0,
                "benign_successes": 0,
                "attack_opportunities": 0,
                "attack_successes": 0,
                "joint_successes": 0,
            }
        )
        for record in completed:
            report = record.get("report", {})
            condition = str(record["condition"])
            key = (condition, str(record["regime"]))
            counts = grouped[key]
            benign_success = bool(report.get("benign_workflow_success"))
            counts["completed"] += 1
            counts["benign_successes"] += benign_success
            if condition == "normal_plus_attack":
                attack_success = bool(report.get("attack_success"))
                counts["attack_opportunities"] += 1
                counts["attack_successes"] += attack_success
                counts["joint_successes"] += benign_success and attack_success

        if status is None:
            status = (
                "completed"
                if len(completed) == len(self.episode_specs)
                else "needs_attention"
                if errors
                else "running"
            )
        by_condition_and_regime = {}
        for (condition, regime), counts in sorted(grouped.items()):
            completed_count = counts["completed"]
            attack_count = counts["attack_opportunities"]
            by_condition_and_regime[f"{condition}__{regime}"] = {
                **counts,
                "benign_success_rate": (
                    counts["benign_successes"] / completed_count
                    if completed_count
                    else None
                ),
                "attack_success_rate": (
                    counts["attack_successes"] / attack_count
                    if attack_count
                    else None
                ),
                "joint_success_rate": (
                    counts["joint_successes"] / attack_count
                    if attack_count
                    else None
                ),
            }
        summary = {
            "schema_version": MIXED_WORKFLOW_BATCH_SCHEMA_VERSION,
            "exp_id": self.exp_id,
            "status": status,
            "total_planned": len(self.episode_specs),
            "completed": len(completed),
            "errors": errors,
            "pending_without_record": len(self.episode_specs) - len(completed) - errors,
            "by_condition_and_regime": by_condition_and_regime,
        }
        _atomic_write_json(self.summary_path, summary)
        _atomic_write_text(
            self.manifest_path,
            (
                "# Paired Mixed Semantic Workflow Experiment\n\n"
                f"- Experiment ID: `{self.exp_id}`\n"
                f"- Status: `{summary['status']}`\n"
                f"- Planned episodes: {summary['total_planned']}\n"
                f"- Completed episodes: {summary['completed']}\n"
                f"- Errors: {summary['errors']}\n"
                "- Command: `command.sh`\n"
                "- Plan: `plan.json`\n"
                "- Summary: `metrics/summary.json`\n"
                "- Episode records: `episodes/`\n"
                "- Logs: `logs/`\n"
            ),
        )
        return summary


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)
