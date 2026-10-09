"""正式目标实例的确定性 Oracle 预检。

预检不使用 LLM、不搜索路径，也不计入正式实验。它只回答：

1. 固定初始状态是否满足目标前提；
2. 固定 action/参数是否被提交给 AgentDojo；
3. 预期状态变化是否真实发生；
4. reset 是否恢复完全相同的初始状态。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .agentdojo_adapter import AgentDojoExecutor
from .agentdojo_verifier import AgentDojoTargetVerifier
from .target_instances import (
    FORMAL_INSTANCES_PREFLIGHT_REPETITIONS,
    FormalTargetInstance,
)


DEFAULT_PREFLIGHT_REPETITIONS = FORMAL_INSTANCES_PREFLIGHT_REPETITIONS


@dataclass(frozen=True, slots=True)
class PreconditionResult:
    ok: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PreflightAttempt:
    repetition: int
    precondition_ok: bool
    precondition_reason: str
    initial_state_consistent: bool
    tool_dispatched: bool
    effect_verified: bool
    reset_verified: bool
    infrastructure_error: bool
    execution_status: str

    @property
    def ready(self) -> bool:
        return all(
            (
                self.precondition_ok,
                self.initial_state_consistent,
                self.tool_dispatched,
                self.effect_verified,
                self.reset_verified,
                not self.infrastructure_error,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "repetition": self.repetition,
            "precondition_ok": self.precondition_ok,
            "precondition_reason": self.precondition_reason,
            "initial_state_consistent": self.initial_state_consistent,
            "tool_dispatched": self.tool_dispatched,
            "effect_verified": self.effect_verified,
            "reset_verified": self.reset_verified,
            "infrastructure_error": self.infrastructure_error,
            "execution_status": self.execution_status,
            "ready": self.ready,
        }


@dataclass(frozen=True, slots=True)
class InstancePreflightResult:
    instance_id: str
    target_type_id: str
    suite_name: str
    action: str
    attempts: tuple[PreflightAttempt, ...]

    @property
    def ready(self) -> bool:
        return bool(self.attempts) and all(
            attempt.ready for attempt in self.attempts
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "target_type_id": self.target_type_id,
            "suite_name": self.suite_name,
            "action": self.action,
            "status": (
                "READY_FOR_FORMAL_RUN"
                if self.ready
                else "INFRASTRUCTURE_ERROR"
            ),
            "attempts": [attempt.as_dict() for attempt in self.attempts],
        }


@dataclass(frozen=True, slots=True)
class FormalPreflightReport:
    benchmark_version: str
    target_manifest_sha256: str
    repetitions: int
    results: tuple[InstancePreflightResult, ...]

    @property
    def ready(self) -> bool:
        return bool(self.results) and all(result.ready for result in self.results)

    def as_dict(self) -> dict[str, Any]:
        ready_count = sum(result.ready for result in self.results)
        return {
            "benchmark_version": self.benchmark_version,
            "target_manifest_sha256": self.target_manifest_sha256,
            "repetitions": self.repetitions,
            "total_instances": len(self.results),
            "total_attempts": sum(
                len(result.attempts) for result in self.results
            ),
            "ready_instances": ready_count,
            "failed_instances": len(self.results) - ready_count,
            "status": (
                "READY_FOR_FORMAL_RUN"
                if self.ready
                else "INFRASTRUCTURE_ERROR"
            ),
            "results": [result.as_dict() for result in self.results],
        }


def check_instance_precondition(
    instance: FormalTargetInstance,
    environment: Any,
) -> PreconditionResult:
    """检查一个固定实例在干净初始状态下是否必然产生新效果。"""

    parameters = dict(instance.parameters)
    action = instance.action

    try:
        if instance.suite_name == "workspace":
            ok, reason = _check_workspace_precondition(
                action,
                parameters,
                environment,
            )
        elif instance.suite_name == "banking":
            ok, reason = _check_banking_precondition(
                action,
                parameters,
                environment,
            )
        else:
            return PreconditionResult(
                False,
                f"UNSUPPORTED_SUITE: {instance.suite_name}",
            )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return PreconditionResult(
            False,
            f"PRECONDITION_INPUT_ERROR: {type(exc).__name__}: {exc}",
        )

    return PreconditionResult(ok, reason)


def run_formal_preflight(
    instances: Iterable[FormalTargetInstance],
    *,
    repetitions: int = DEFAULT_PREFLIGHT_REPETITIONS,
) -> FormalPreflightReport:
    """对全部固定实例运行重复的 reset→execute→verify→reset 检查。"""

    if repetitions <= 0:
        raise ValueError("预检重复次数必须是正整数。")

    materialized = tuple(instances)
    executors = {
        suite_name: AgentDojoExecutor(suite_name)
        for suite_name in {instance.suite_name for instance in materialized}
    }
    verifier = AgentDojoTargetVerifier()
    results: list[InstancePreflightResult] = []

    for instance in materialized:
        executor = executors[instance.suite_name]
        reference_initial_fingerprint: str | None = None
        attempts: list[PreflightAttempt] = []

        for repetition in range(1, repetitions + 1):
            executor.reset()
            initial_fingerprint = _state_fingerprint(
                verifier,
                instance,
                executor,
            )
            if reference_initial_fingerprint is None:
                reference_initial_fingerprint = initial_fingerprint
            initial_state_consistent = (
                initial_fingerprint == reference_initial_fingerprint
            )

            precondition = check_instance_precondition(
                instance,
                executor.backend_environment,
            )

            if precondition.ok and initial_state_consistent:
                execution = executor.execute(
                    "ORACLE_PREFLIGHT_AGENT",
                    instance.action,
                    instance.parameters,
                )
                tool_dispatched = execution.tool_dispatched
                effect_verified = execution.effect_verified is True
                infrastructure_error = execution.infrastructure_error
                execution_status = execution.message.split(":", 1)[0]
            else:
                tool_dispatched = False
                effect_verified = False
                infrastructure_error = True
                execution_status = "PRECONDITION_FAILED"

            executor.reset()
            reset_fingerprint = _state_fingerprint(
                verifier,
                instance,
                executor,
            )
            reset_verified = reset_fingerprint == reference_initial_fingerprint

            attempts.append(
                PreflightAttempt(
                    repetition=repetition,
                    precondition_ok=precondition.ok,
                    precondition_reason=precondition.reason,
                    initial_state_consistent=initial_state_consistent,
                    tool_dispatched=tool_dispatched,
                    effect_verified=effect_verified,
                    reset_verified=reset_verified,
                    infrastructure_error=infrastructure_error,
                    execution_status=execution_status,
                )
            )

        results.append(
            InstancePreflightResult(
                instance_id=instance.instance_id,
                target_type_id=instance.target_type_id,
                suite_name=instance.suite_name,
                action=instance.action,
                attempts=tuple(attempts),
            )
        )

    return FormalPreflightReport(
        benchmark_version="v1.2.2",
        target_manifest_sha256=_instances_fingerprint(materialized),
        repetitions=repetitions,
        results=tuple(results),
    )


def _state_fingerprint(
    verifier: AgentDojoTargetVerifier,
    instance: FormalTargetInstance,
    executor: AgentDojoExecutor,
) -> str:
    state = verifier.capture_if_formal(
        instance.suite_name,
        instance.action,
        executor.backend_environment,
    )
    if state is None:
        raise ValueError(f"{instance.instance_id} 不是正式目标动作。")
    canonical = json.dumps(
        state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _instances_fingerprint(
    instances: tuple[FormalTargetInstance, ...],
) -> str:
    """绑定实例 ID、工具和全部参数，防止清单变化后复用旧报告。"""

    manifest = [
        {
            "instance_id": instance.instance_id,
            "target_type_id": instance.target_type_id,
            "suite_name": instance.suite_name,
            "action": instance.action,
            "description": instance.description,
            "parameters": dict(instance.parameters),
        }
        for instance in instances
    ]
    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _check_workspace_precondition(
    action: str,
    parameters: Mapping[str, Any],
    environment: Any,
) -> tuple[bool, str]:
    inbox = environment.inbox
    calendar = environment.calendar
    cloud_drive = environment.cloud_drive

    if action == "send_email":
        recipients = list(parameters["recipients"])
        if not recipients:
            return False, "RECIPIENTS_EMPTY"
        duplicate_exists = any(
            email.status.value == "sent"
            and email.recipients == recipients
            and email.subject == parameters["subject"]
            and email.body == parameters["body"]
            for email in inbox.emails.values()
        )
        return (
            (not duplicate_exists),
            "NEW_SENT_EMAIL_WILL_BE_CREATED"
            if not duplicate_exists
            else "TARGET_EMAIL_ALREADY_EXISTS",
        )

    if action == "delete_email":
        email_id = str(parameters["email_id"])
        ok = email_id in inbox.emails and email_id not in inbox.trash
        return ok, "EMAIL_EXISTS" if ok else "EMAIL_NOT_DELETABLE"

    if action == "create_calendar_event":
        start = _parse_calendar_time(parameters["start_time"])
        end = _parse_calendar_time(parameters["end_time"])
        if end <= start:
            return False, "EVENT_END_NOT_AFTER_START"
        duplicate_exists = any(
            event.title == parameters["title"]
            and event.start_time == start
            and event.end_time == end
            for event in calendar.events.values()
        )
        return (
            (not duplicate_exists),
            "NEW_EVENT_WILL_BE_CREATED"
            if not duplicate_exists
            else "TARGET_EVENT_ALREADY_EXISTS",
        )

    if action == "cancel_calendar_event":
        event_id = str(parameters["event_id"])
        event = calendar.events.get(event_id)
        ok = event is not None and event.status.value != "canceled"
        return ok, "EVENT_IS_CONFIRMED" if ok else "EVENT_NOT_CANCELABLE"

    if action == "reschedule_calendar_event":
        event_id = str(parameters["event_id"])
        event = calendar.events.get(event_id)
        if event is None:
            return False, "EVENT_NOT_FOUND"
        new_start = _parse_calendar_time(parameters["new_start_time"])
        new_end_raw = parameters.get("new_end_time")
        if new_end_raw is None:
            new_end = new_start + (event.end_time - event.start_time)
        else:
            new_end = _parse_calendar_time(new_end_raw)
        if new_end <= new_start:
            return False, "EVENT_END_NOT_AFTER_START"
        changed = new_start != event.start_time or new_end != event.end_time
        return changed, "EVENT_TIME_WILL_CHANGE" if changed else "EVENT_TIME_NOOP"

    if action == "add_calendar_event_participants":
        event_id = str(parameters["event_id"])
        event = calendar.events.get(event_id)
        requested = set(parameters["participants"])
        if event is None:
            return False, "EVENT_NOT_FOUND"
        if not requested:
            return False, "PARTICIPANTS_EMPTY"
        new_participants = requested - set(event.participants)
        ok = new_participants == requested
        return (
            ok,
            "ALL_PARTICIPANTS_ARE_NEW"
            if ok
            else "PARTICIPANT_ALREADY_PRESENT",
        )

    if action == "create_file":
        filename = str(parameters["filename"])
        content = str(parameters["content"])
        if not filename or not content:
            return False, "FILENAME_OR_CONTENT_EMPTY"
        duplicate_exists = any(
            file.filename == filename for file in cloud_drive.files.values()
        )
        return (
            (not duplicate_exists),
            "NEW_FILE_WILL_BE_CREATED"
            if not duplicate_exists
            else "FILENAME_ALREADY_EXISTS",
        )

    if action == "append_to_file":
        file_id = str(parameters["file_id"])
        content = str(parameters["content"])
        ok = file_id in cloud_drive.files and bool(content)
        return ok, "FILE_EXISTS_AND_APPEND_NONEMPTY" if ok else "APPEND_INVALID"

    if action == "delete_file":
        file_id = str(parameters["file_id"])
        ok = file_id in cloud_drive.files
        return ok, "FILE_EXISTS" if ok else "FILE_NOT_FOUND"

    if action == "share_file":
        file_id = str(parameters["file_id"])
        file = cloud_drive.files.get(file_id)
        if file is None:
            return False, "FILE_NOT_FOUND"
        current_permission = file.shared_with.get(parameters["email"])
        if current_permission is not None:
            current_permission = current_permission.value
        changed = current_permission != parameters["permission"]
        return changed, "SHARING_PERMISSION_WILL_CHANGE" if changed else "SHARE_NOOP"

    return False, f"UNSUPPORTED_WORKSPACE_ACTION: {action}"


def _check_banking_precondition(
    action: str,
    parameters: Mapping[str, Any],
    environment: Any,
) -> tuple[bool, str]:
    bank_account = environment.bank_account
    user_account = environment.user_account

    if action == "send_money":
        if parameters["amount"] <= 0:
            return False, "AMOUNT_NOT_POSITIVE"
        duplicate_exists = any(
            transaction.recipient == parameters["recipient"]
            and transaction.amount == parameters["amount"]
            and transaction.subject == parameters["subject"]
            and transaction.date == parameters["date"]
            for transaction in bank_account.transactions
        )
        return (
            (not duplicate_exists),
            "NEW_TRANSACTION_WILL_BE_CREATED"
            if not duplicate_exists
            else "TARGET_TRANSACTION_ALREADY_EXISTS",
        )

    if action == "schedule_transaction":
        if parameters["amount"] <= 0:
            return False, "AMOUNT_NOT_POSITIVE"
        duplicate_exists = any(
            transaction.recipient == parameters["recipient"]
            and transaction.amount == parameters["amount"]
            and transaction.subject == parameters["subject"]
            and transaction.date == parameters["date"]
            and transaction.recurring == parameters["recurring"]
            for transaction in bank_account.scheduled_transactions
        )
        return (
            (not duplicate_exists),
            "NEW_SCHEDULED_TRANSACTION_WILL_BE_CREATED"
            if not duplicate_exists
            else "TARGET_SCHEDULE_ALREADY_EXISTS",
        )

    if action == "update_scheduled_transaction":
        transaction = next(
            (
                item
                for item in bank_account.scheduled_transactions
                if item.id == parameters["id"]
            ),
            None,
        )
        if transaction is None:
            return False, "SCHEDULED_TRANSACTION_NOT_FOUND"
        mutable_fields = ("recipient", "amount", "subject", "date", "recurring")
        requested = {
            field: parameters[field]
            for field in mutable_fields
            if field in parameters and parameters[field] is not None
        }
        if not requested:
            return False, "NO_UPDATE_FIELD_PROVIDED"
        # AgentDojo v1.2.2 使用 if recurring，因此 True→False 无法应用。
        if requested.get("recurring") is False:
            return False, "AGENTDOJO_V122_FALSE_RECURRING_UPDATE_UNSUPPORTED"
        for field, value in requested.items():
            if field != "recurring" and not value:
                return False, f"FALSY_UPDATE_VALUE_UNSUPPORTED: {field}"
        changed = any(
            getattr(transaction, field) != value
            for field, value in requested.items()
        )
        return changed, "SCHEDULED_TRANSACTION_WILL_CHANGE" if changed else "UPDATE_NOOP"

    if action == "update_password":
        password = str(parameters["password"])
        changed = bool(password) and password != user_account.password
        return changed, "PASSWORD_WILL_CHANGE" if changed else "PASSWORD_NOOP"

    if action == "update_user_info":
        mutable_fields = ("first_name", "last_name", "street", "city")
        requested = {
            field: parameters[field]
            for field in mutable_fields
            if field in parameters and parameters[field] is not None
        }
        if not requested or any(not value for value in requested.values()):
            return False, "USER_INFO_UPDATE_EMPTY"
        changed = any(
            getattr(user_account, field) != value
            for field, value in requested.items()
        )
        return changed, "USER_INFO_WILL_CHANGE" if changed else "USER_INFO_NOOP"

    return False, f"UNSUPPORTED_BANKING_ACTION: {action}"


def _parse_calendar_time(value: str):
    import datetime as dt

    return dt.datetime.strptime(value, "%Y-%m-%d %H:%M")
