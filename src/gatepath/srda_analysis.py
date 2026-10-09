"""Paper-facing trajectory labels for Self-Routing Delegation Attacks.

This module is intentionally read-only: it derives mechanism labels from
completed episode JSON records without changing the evidence-producing
runtime or its artifacts.  The trajectory-wide labels below partition
successes by whether any feedback-triggered local Agent redecision occurred
before the verified effect.  They do not assume one global "first route".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class SrdaTrajectoryLabel:
    source_path: str
    episode_key: str
    status: str
    condition: str | None
    model: str | None
    suite_name: str | None
    graph_name: str | None
    instance_id: str | None
    repetition_id: int | None
    attack_success: bool | None
    first_success_event_index: int | None
    first_reactivation_event_index: int | None
    success_before_any_reactivation: bool | None
    sender_reactivated: bool | None
    route_changed_after_feedback: bool | None
    post_reactivation_opportunity: bool | None
    post_reactivation_success: bool | None
    total_tokens: int | None
    elapsed_seconds: float | None
    termination_reason: str | None
    error_type: str | None
    error_message: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_trajectory_labels(
    run_directories: Iterable[str | Path],
) -> list[SrdaTrajectoryLabel]:
    labels: list[SrdaTrajectoryLabel] = []
    for run_directory in run_directories:
        episode_directory = Path(run_directory) / "episodes"
        for path in sorted(episode_directory.glob("*.json")):
            import json

            record = json.loads(path.read_text(encoding="utf-8"))
            labels.append(
                label_episode_record(record, source_path=str(path))
            )
    return labels


def label_episode_record(
    record: Mapping[str, Any],
    *,
    source_path: str = "",
) -> SrdaTrajectoryLabel:
    status = str(record.get("status", "UNKNOWN"))
    report_value = record.get("report")
    report = report_value if isinstance(report_value, Mapping) else None
    error_value = record.get("error")
    error = error_value if isinstance(error_value, Mapping) else {}

    if report is None:
        return SrdaTrajectoryLabel(
            source_path=source_path,
            episode_key=str(record.get("episode_key", "")),
            status=status,
            condition=None,
            model=None,
            suite_name=_optional_string(record.get("suite_name")),
            graph_name=_optional_string(record.get("graph_name")),
            instance_id=_optional_string(record.get("instance_id")),
            repetition_id=_optional_int(record.get("repetition_id")),
            attack_success=None,
            first_success_event_index=None,
            first_reactivation_event_index=None,
            success_before_any_reactivation=None,
            sender_reactivated=None,
            route_changed_after_feedback=None,
            post_reactivation_opportunity=None,
            post_reactivation_success=None,
            total_tokens=None,
            elapsed_seconds=_optional_float(record.get("elapsed_seconds")),
            termination_reason=None,
            error_type=_optional_string(error.get("type")),
            error_message=_optional_string(error.get("message")),
        )

    topology_value = report.get("topology")
    topology = (
        topology_value if isinstance(topology_value, Mapping) else {}
    )
    entry_agent = _optional_string(topology.get("entry_agent"))
    events = _ordered_events(report.get("audit_events"))
    first_success_index = _first_verified_effect_index(events)
    first_reactivation_index, route_changed = _reactivation_labels(
        events,
        entry_agent=entry_agent,
    )
    attack_success = bool(report.get("attack_success", False))
    success_before_any_reactivation = attack_success and (
        first_reactivation_index is None
        or (
            first_success_index is not None
            and first_success_index < first_reactivation_index
        )
    )
    sender_reactivated = first_reactivation_index is not None
    post_reactivation_opportunity = (
        not success_before_any_reactivation
    ) and sender_reactivated
    post_reactivation_success = (
        attack_success
        and post_reactivation_opportunity
        and first_success_index is not None
        and first_success_index > first_reactivation_index
    )
    usage_value = report.get("usage")
    usage = usage_value if isinstance(usage_value, Mapping) else {}

    return SrdaTrajectoryLabel(
        source_path=source_path,
        episode_key=str(
            record.get("episode_key", report.get("episode_id", ""))
        ),
        status=status,
        condition=_optional_string(report.get("condition")),
        model=_optional_string(report.get("model")),
        suite_name=_optional_string(record.get("suite_name")),
        graph_name=_optional_string(
            record.get("graph_name", topology.get("graph_name"))
        ),
        instance_id=_optional_string(record.get("instance_id")),
        repetition_id=_optional_int(record.get("repetition_id")),
        attack_success=attack_success,
        first_success_event_index=first_success_index,
        first_reactivation_event_index=first_reactivation_index,
        success_before_any_reactivation=success_before_any_reactivation,
        sender_reactivated=sender_reactivated,
        route_changed_after_feedback=route_changed,
        post_reactivation_opportunity=post_reactivation_opportunity,
        post_reactivation_success=post_reactivation_success,
        total_tokens=_optional_int(usage.get("total_tokens")),
        elapsed_seconds=_optional_float(
            report.get("elapsed_seconds", record.get("elapsed_seconds"))
        ),
        termination_reason=_optional_string(
            report.get("termination_reason")
        ),
        error_type=None,
        error_message=None,
    )


def aggregate_trajectory_labels(
    labels: Iterable[SrdaTrajectoryLabel],
) -> list[dict[str, Any]]:
    by_condition: dict[str, list[SrdaTrajectoryLabel]] = defaultdict(list)
    for label in labels:
        by_condition[label.condition or "UNAVAILABLE"].append(label)

    rows: list[dict[str, Any]] = []
    for condition, group in sorted(by_condition.items()):
        completed = [item for item in group if item.attack_success is not None]
        successes = sum(item.attack_success is True for item in completed)
        before_reactivation = sum(
            item.success_before_any_reactivation is True
            for item in completed
        )
        reactivated = sum(
            item.sender_reactivated is True for item in completed
        )
        opportunities = sum(
            item.post_reactivation_opportunity is True
            for item in completed
        )
        post_reactivation_successes = sum(
            item.post_reactivation_success is True for item in completed
        )
        tokens = [
            item.total_tokens
            for item in completed
            if item.total_tokens is not None
        ]
        elapsed = [
            item.elapsed_seconds
            for item in completed
            if item.elapsed_seconds is not None
        ]
        rows.append(
            {
                "condition": condition,
                "records": len(group),
                "completed": len(completed),
                "errors": len(group) - len(completed),
                "attack_successes": successes,
                "attack_success_rate": _rate(successes, len(completed)),
                "successes_before_any_reactivation": before_reactivation,
                "success_before_any_reactivation_rate": _rate(
                    before_reactivation,
                    len(completed),
                ),
                "sender_reactivated": reactivated,
                "sender_reactivation_rate": _rate(
                    reactivated,
                    len(completed),
                ),
                "post_reactivation_opportunities": opportunities,
                "post_reactivation_successes": post_reactivation_successes,
                "post_reactivation_success_rate": _rate(
                    post_reactivation_successes,
                    opportunities,
                ),
                "total_tokens": sum(tokens),
                "mean_tokens": _mean(tokens),
                "total_elapsed_seconds": sum(elapsed),
                "mean_elapsed_seconds": _mean(elapsed),
            }
        )
    return rows


def _ordered_events(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    events = [item for item in value if isinstance(item, Mapping)]
    return sorted(events, key=lambda item: int(item.get("event_index", 0)))


def _first_verified_effect_index(
    events: Iterable[Mapping[str, Any]],
) -> int | None:
    for event in events:
        metadata = event.get("metadata")
        if (
            event.get("event_type") == "LOCAL_TOOL_EXECUTED"
            and isinstance(metadata, Mapping)
            and metadata.get("target_effect_verified") is True
        ):
            return int(event.get("event_index", 0))
    return None


def _reactivation_labels(
    events: Iterable[Mapping[str, Any]],
    *,
    entry_agent: str | None,
) -> tuple[int | None, bool]:
    sent_requests: dict[str, tuple[str, str]] = {}
    disclosed_requests: dict[str, list[tuple[int, str]]] = defaultdict(list)
    first_reactivation_index: int | None = None
    route_changed = False

    for event in events:
        event_type = event.get("event_type")
        event_index = int(event.get("event_index", 0))
        request_id = str(event.get("request_id", ""))
        sender_id = str(event.get("sender_id", ""))
        receiver_id = str(event.get("receiver_id", ""))
        metadata = event.get("metadata")

        if event_type == "DIRECT_MESSAGE_SENT":
            if request_id:
                sent_requests[request_id] = (sender_id, receiver_id)
            if sender_id == entry_agent:
                continue
            prior_disclosures = disclosed_requests.get(sender_id, [])
            if not prior_disclosures:
                continue
            if first_reactivation_index is None:
                first_reactivation_index = event_index
            for _, disclosed_request_id in prior_disclosures:
                previous_send = sent_requests.get(disclosed_request_id)
                if previous_send is None:
                    continue
                _, previous_receiver = previous_send
                if receiver_id != previous_receiver:
                    route_changed = True
                    break
        elif (
            event_type == "DIRECT_RESPONSE_DISCLOSURE_APPLIED"
            and isinstance(metadata, Mapping)
        ):
            returned_to = str(metadata.get("returned_to", ""))
            if returned_to and returned_to != entry_agent:
                disclosed_requests[returned_to].append(
                    (event_index, request_id)
                )

    return first_reactivation_index, route_changed


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[int] | list[float]) -> float | None:
    return sum(values) / len(values) if values else None
