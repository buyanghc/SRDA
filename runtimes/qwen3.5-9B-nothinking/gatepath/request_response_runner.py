"""真实AutoGen请求—响应episode的构建、运行与最终评测。"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from autogen_core.models import ChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient

from .agentdojo_adapter import AgentDojoExecutor
from .agentdojo_verifier import AgentDojoTargetVerifier
from .deepseek_model_client import DeepSeekChatCompletionClient
from .experiment_graphs import ExperimentTopology
from .preflight import PreconditionResult, check_instance_precondition
from .request_response_protocol import (
    REQUEST_RESPONSE_SCHEMA_VERSION,
    DirectTeamResponse,
    EntryAgentMode,
    HandoffContinuation,
    NormalAgentTrace,
    RequestResponseAuditEvent,
    RequestResponseBudget,
    RequestResponsePromptBundle,
    ResponseVisibility,
    build_attack_request_content,
)
from .request_response_runtime import (
    AutoGenRequestResponseRuntime,
    RequestResponseBudgetExhausted,
    RequestResponseRecorder,
)
from .role_assignment import RoleAssignedWorld, build_role_assigned_world
from .target_instances import FormalTargetInstance


@dataclass(frozen=True, slots=True)
class RequestResponseEpisodeReport:
    episode_id: str
    condition: str
    entry_agent_mode: EntryAgentMode
    response_visibility: ResponseVisibility
    handoff_continuation: HandoffContinuation
    topology: ExperimentTopology
    assigned_team: RoleAssignedWorld
    target: FormalTargetInstance
    prompt_version: str
    prompt_language: str
    model: str
    budget: RequestResponseBudget
    precondition: PreconditionResult
    termination_reason: str
    attack_success: bool
    target_agent_reached: bool
    target_tool_dispatched: bool
    target_tool_dispatch_count: int
    target_effect_verified: bool
    sensitive_tool_dispatch_count: int
    repeated_sensitive_tool_dispatch_count: int
    sensitive_tool_dispatches: tuple[Mapping[str, Any], ...]
    off_target_sensitive_tool_executed: bool
    off_target_sensitive_tool_execution_count: int
    off_target_sensitive_tool_executions: tuple[Mapping[str, Any], ...]
    infrastructure_error: bool
    message_budget_exhausted: bool
    runtime_budget_exhausted: bool
    blocked_message_attempts: tuple[Mapping[str, Any], ...]
    messages_sent: int
    agents_reached: tuple[str, ...]
    node_visit_counts: Mapping[str, int]
    repeated_node_visit_count: int
    edge_traversal_counts: tuple[Mapping[str, Any], ...]
    repeated_edge_traversal_count: int
    cycle_closing_message_count: int
    cycle_closing_request_ids: tuple[str, ...]
    successful_route: tuple[str, ...] | None
    entry_responses: tuple[DirectTeamResponse, ...]
    entry_final_response: DirectTeamResponse | None
    elapsed_seconds: float
    usage: Mapping[str, int]
    traces: tuple[NormalAgentTrace, ...]
    audit_events: tuple[RequestResponseAuditEvent, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REQUEST_RESPONSE_SCHEMA_VERSION,
            "experiment_type": "AUTOGEN_NESTED_REQUEST_RESPONSE",
            "episode_id": self.episode_id,
            "condition": self.condition,
            "entry_agent_mode": self.entry_agent_mode.value,
            "response_visibility": self.response_visibility.value,
            "handoff_continuation": self.handoff_continuation.value,
            "topology": self.topology.as_dict(),
            "assigned_team": self.assigned_team.as_dict(),
            "target": {
                "instance_id": self.target.instance_id,
                "target_type_id": self.target.target_type_id,
                "suite_name": self.target.suite_name,
                "action": self.target.action,
                "description": self.target.description,
                "parameters": dict(self.target.parameters),
            },
            "prompt": {
                "version": self.prompt_version,
                "language": self.prompt_language,
            },
            "model": self.model,
            "budget": self.budget.as_dict(),
            "precondition": {
                "ok": self.precondition.ok,
                "reason": self.precondition.reason,
            },
            "termination_reason": self.termination_reason,
            "attack_success": self.attack_success,
            "target_agent_reached": self.target_agent_reached,
            "target_tool_dispatched": self.target_tool_dispatched,
            "target_tool_dispatch_count": self.target_tool_dispatch_count,
            "target_effect_verified": self.target_effect_verified,
            "sensitive_tool_dispatch_count": (
                self.sensitive_tool_dispatch_count
            ),
            "repeated_sensitive_tool_dispatch_count": (
                self.repeated_sensitive_tool_dispatch_count
            ),
            "sensitive_tool_dispatches": [
                dict(dispatch) for dispatch in self.sensitive_tool_dispatches
            ],
            "off_target_sensitive_tool_executed": (
                self.off_target_sensitive_tool_executed
            ),
            "off_target_sensitive_tool_execution_count": (
                self.off_target_sensitive_tool_execution_count
            ),
            "off_target_sensitive_tool_executions": [
                dict(execution)
                for execution in self.off_target_sensitive_tool_executions
            ],
            "infrastructure_error": self.infrastructure_error,
            "message_budget_exhausted": self.message_budget_exhausted,
            "runtime_budget_exhausted": self.runtime_budget_exhausted,
            "blocked_message_attempts": [
                dict(attempt) for attempt in self.blocked_message_attempts
            ],
            "messages_sent": self.messages_sent,
            "agents_reached": list(self.agents_reached),
            "node_visit_counts": dict(self.node_visit_counts),
            "repeated_node_visit_count": self.repeated_node_visit_count,
            "edge_traversal_counts": [
                dict(edge) for edge in self.edge_traversal_counts
            ],
            "repeated_edge_traversal_count": (
                self.repeated_edge_traversal_count
            ),
            "cycle_closing_message_count": (
                self.cycle_closing_message_count
            ),
            "cycle_closing_request_ids": list(
                self.cycle_closing_request_ids
            ),
            "successful_route": (
                None
                if self.successful_route is None
                else list(self.successful_route)
            ),
            "entry_responses": [
                {
                    "request_id": response.request_id,
                    "responder_id": response.responder_id,
                    "content": response.content,
                }
                for response in self.entry_responses
            ],
            "entry_final_response": (
                None
                if self.entry_final_response is None
                else {
                    "request_id": self.entry_final_response.request_id,
                    "responder_id": self.entry_final_response.responder_id,
                    "content": self.entry_final_response.content,
                }
            ),
            "elapsed_seconds": self.elapsed_seconds,
            "usage": dict(self.usage),
            "traces": [trace.as_dict() for trace in self.traces],
            "audit_events": [
                event.as_dict() for event in self.audit_events
            ],
        }


async def run_request_response_episode_async(
    *,
    instance: FormalTargetInstance,
    topology: ExperimentTopology,
    prompts: RequestResponsePromptBundle,
    condition: str = "CLEAN_RAW_RESPONSE_NO_FEEDBACK_DISCLOSURE_TRANSFORM",
    budget: RequestResponseBudget | None = None,
    episode_id: str | None = None,
    executor: AgentDojoExecutor | None = None,
    role_seed: int = 0,
    model: str = "Qwen3-14B",
    base_url: str = "http://127.0.0.1:8010/v1",
    api_key: str = "EMPTY",
    model_client: ChatCompletionClient | None = None,
    temperature: float = 0.0,
    thinking_mode: str | None = None,
    reasoning_effort: str | None = None,
    entry_agent_mode: EntryAgentMode | str = EntryAgentMode.DETERMINISTIC,
    response_visibility: ResponseVisibility | str = (
        ResponseVisibility.NATURAL
    ),
    handoff_continuation: HandoffContinuation | str = (
        HandoffContinuation.RETURN_TO_SENDER
    ),
    assigned_team_override: RoleAssignedWorld | None = None,
    request_content_override: str | None = None,
    reset_executor: bool = True,
) -> RequestResponseEpisodeReport:
    """运行可配置入口Agent与下游回复可见性的嵌套episode。"""

    active_budget = budget or RequestResponseBudget()
    active_entry_mode = EntryAgentMode(entry_agent_mode)
    active_response_visibility = ResponseVisibility(response_visibility)
    active_handoff_continuation = HandoffContinuation(
        handoff_continuation
    )
    if topology.target_agent is None:
        raise ValueError("请求—响应第一阶段不运行G0。")
    if episode_id is None:
        episode_id = (
            f"request-response-{topology.graph_id}-"
            f"{instance.instance_id.lower()}-{uuid.uuid4().hex[:8]}"
        )

    active_executor = executor or AgentDojoExecutor(instance.suite_name)
    if active_executor.suite_name != instance.suite_name:
        raise ValueError("executor的suite与目标实例不一致。")
    if reset_executor:
        active_executor.reset()

    precondition = check_instance_precondition(
        instance,
        active_executor.backend_environment,
    )
    if not precondition.ok:
        raise RuntimeError(
            "REQUEST_RESPONSE_PREFLIGHT_FAILED: "
            f"{instance.instance_id}: {precondition.reason}"
        )

    verifier = AgentDojoTargetVerifier()
    before_state = verifier.capture_if_formal(
        instance.suite_name,
        instance.action,
        active_executor.backend_environment,
    )
    if before_state is None:
        raise RuntimeError(
            f"{instance.instance_id}不是GatePath正式目标动作。"
        )

    if assigned_team_override is None:
        assigned_team = build_role_assigned_world(
            topology,
            suite_name=instance.suite_name,
            target_action=instance.action,
            role_seed=role_seed,
            available_tools=active_executor.available_actions,
        )
    else:
        assigned_team = assigned_team_override
        _validate_assigned_team_override(
            assigned_team=assigned_team,
            topology=topology,
            instance=instance,
            available_tools=active_executor.available_actions,
        )
    world = assigned_team.world
    content = request_content_override or build_attack_request_content(
        action=instance.action,
        arguments=instance.parameters,
        description=instance.description,
        workflow_note=prompts.workflow_note,
        request_preamble=prompts.request_preamble,
    )
    recorder = RequestResponseRecorder(
        episode_id=episode_id,
        budget=active_budget,
        executor=active_executor,
        verifier=verifier,
        target_action=instance.action,
        target_arguments=instance.parameters,
        target_before_state=before_state,
    )
    root_request_id = f"{episode_id}-ROOT"
    if prompts.preserve_canonical_work_item:
        recorder.register_canonical_work_item(
            root_request_id=root_request_id,
            content=content,
        )

    owns_model_client = model_client is None
    active_model_client = model_client or _build_model_client(
        **_openai_model_client_kwargs(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        ),
        thinking_mode=thinking_mode,
        reasoning_effort=reasoning_effort,
    )
    client_usage_before = _usage_from_model_client(active_model_client)
    client_usage_after = client_usage_before
    started_at = time.monotonic()
    termination_reason = (
        "ENTRY_AGENT_COMPLETED"
        if active_entry_mode is EntryAgentMode.LLM
        else "ENTRY_RESPONSES_RETURNED"
    )
    infrastructure_error = False
    entry_responses: tuple[DirectTeamResponse, ...] = ()
    try:
        async with AutoGenRequestResponseRuntime(
            world=world,
            entry_agent_id=topology.entry_agent,
            model_client=active_model_client,
            executor=active_executor,
            prompts=prompts,
            recorder=recorder,
            max_tool_iterations=active_budget.max_tool_iterations,
            entry_agent_mode=active_entry_mode,
            response_visibility=active_response_visibility,
            handoff_continuation=active_handoff_continuation,
        ) as runtime:
            entry_responses = await asyncio.wait_for(
                runtime.send_from_compromised_entry(
                    root_request_id=root_request_id,
                    content=content,
                ),
                timeout=active_budget.max_runtime_seconds,
            )
    except asyncio.TimeoutError:
        termination_reason = "RUNTIME_BUDGET_EXHAUSTED"
        recorder.runtime_budget_exhausted = True
    except RequestResponseBudgetExhausted as exc:
        termination_reason = exc.reason
    except Exception:
        termination_reason = "INFRASTRUCTURE_ERROR"
        infrastructure_error = True
        raise
    finally:
        client_usage_after = _usage_from_model_client(active_model_client)
        if owns_model_client:
            await active_model_client.close()

    final_verification = verifier.verify_if_formal(
        instance.suite_name,
        instance.action,
        instance.parameters,
        before_state,
        active_executor.backend_environment,
    )
    if final_verification is None:
        raise RuntimeError("正式目标未获得状态验证结果。")
    attack_success = _attack_succeeded_during_episode(
        observed_effect=recorder.target_effect_verified,
        final_effect=final_verification.succeeded,
    )
    reached_order = _reached_agents(recorder)
    usage = _merge_usage(
        _usage_delta(client_usage_before, client_usage_after),
        _usage_from_traces(recorder.traces),
    )
    target_agent_reached = topology.target_agent in reached_order
    elapsed_seconds = time.monotonic() - started_at
    if recorder.message_budget_exhausted:
        termination_reason = "MESSAGE_BUDGET_EXHAUSTED"
    elif recorder.runtime_budget_exhausted:
        termination_reason = "RUNTIME_BUDGET_EXHAUSTED"
    route_metrics = recorder.route_metrics()

    return RequestResponseEpisodeReport(
        episode_id=episode_id,
        condition=condition,
        entry_agent_mode=active_entry_mode,
        response_visibility=active_response_visibility,
        handoff_continuation=active_handoff_continuation,
        topology=topology,
        assigned_team=assigned_team,
        target=instance,
        prompt_version=prompts.version,
        prompt_language=prompts.language.value,
        model=model,
        budget=active_budget,
        precondition=precondition,
        termination_reason=termination_reason,
        attack_success=attack_success,
        target_agent_reached=target_agent_reached,
        target_tool_dispatched=recorder.target_tool_dispatched,
        target_tool_dispatch_count=recorder.target_tool_dispatch_count,
        target_effect_verified=attack_success,
        sensitive_tool_dispatch_count=len(
            recorder.sensitive_tool_dispatches
        ),
        repeated_sensitive_tool_dispatch_count=(
            recorder.repeated_sensitive_tool_dispatch_count()
        ),
        sensitive_tool_dispatches=tuple(
            dict(dispatch) for dispatch in recorder.sensitive_tool_dispatches
        ),
        off_target_sensitive_tool_executed=bool(
            recorder.off_target_sensitive_tool_executions
        ),
        off_target_sensitive_tool_execution_count=len(
            recorder.off_target_sensitive_tool_executions
        ),
        off_target_sensitive_tool_executions=tuple(
            dict(execution)
            for execution in recorder.off_target_sensitive_tool_executions
        ),
        infrastructure_error=(
            infrastructure_error or recorder.infrastructure_error
        ),
        message_budget_exhausted=recorder.message_budget_exhausted,
        runtime_budget_exhausted=recorder.runtime_budget_exhausted,
        blocked_message_attempts=tuple(
            dict(attempt) for attempt in recorder.blocked_message_attempts
        ),
        messages_sent=recorder.message_count,
        agents_reached=reached_order,
        node_visit_counts=dict(route_metrics["node_visit_counts"]),
        repeated_node_visit_count=int(
            route_metrics["repeated_node_visit_count"]
        ),
        edge_traversal_counts=tuple(
            dict(edge) for edge in route_metrics["edge_traversal_counts"]
        ),
        repeated_edge_traversal_count=int(
            route_metrics["repeated_edge_traversal_count"]
        ),
        cycle_closing_message_count=int(
            route_metrics["cycle_closing_message_count"]
        ),
        cycle_closing_request_ids=tuple(
            str(item)
            for item in route_metrics["cycle_closing_request_ids"]
        ),
        successful_route=recorder.successful_route(),
        entry_responses=entry_responses,
        entry_final_response=recorder.entry_final_response,
        elapsed_seconds=elapsed_seconds,
        usage=usage,
        traces=tuple(recorder.traces),
        audit_events=tuple(recorder.events),
    )


def _validate_assigned_team_override(
    *,
    assigned_team: RoleAssignedWorld,
    topology: ExperimentTopology,
    instance: FormalTargetInstance,
    available_tools: tuple[str, ...] | frozenset[str] | set[str],
) -> None:
    """Validate a fixed semantic role placement without random reassignment."""

    if assigned_team.suite_name != instance.suite_name:
        raise ValueError("固定角色团队与目标实例的suite不一致。")
    if assigned_team.target_action != instance.action:
        raise ValueError("固定角色团队与目标实例的target action不一致。")
    if assigned_team.target_agent != topology.target_agent:
        raise ValueError("固定角色团队与拓扑的target agent不一致。")
    if set(assigned_team.world.agent_ids) != set(topology.nodes):
        raise ValueError("固定角色团队的节点集合与拓扑不一致。")

    available = frozenset(available_tools)
    target_holders = []
    for agent_id in topology.nodes:
        spec = assigned_team.world.spec_for(agent_id)
        if spec.contacts != topology.contacts_for(agent_id):
            raise ValueError(f"固定角色 {agent_id!r} 的联系人与拓扑不一致。")
        unavailable = spec.capabilities - available
        if unavailable:
            raise ValueError(
                f"固定角色 {agent_id!r} 使用了suite中不存在的工具："
                f"{sorted(unavailable)}"
            )
        if instance.action in spec.capabilities:
            target_holders.append(agent_id)
    if target_holders != [topology.target_agent]:
        raise ValueError(
            "固定角色团队中必须只有target agent持有目标动作；"
            f"holders={target_holders}。"
        )


def run_request_response_episode(
    **kwargs: Any,
) -> RequestResponseEpisodeReport:
    return asyncio.run(run_request_response_episode_async(**kwargs))


def _openai_model_client_kwargs(
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
) -> dict[str, Any]:
    """Build OpenAI-compatible client arguments without leaking credentials.

    The local Qwen server needs its model-specific chat-template switch.  A
    hosted OpenAI-compatible endpoint such as OpenRouter must not receive that
    vLLM/Qwen-only request body.  Ministral's message schema does not accept
    OpenAI ``name`` fields, so AutoGen preserves the same sender identity as a
    content prefix instead.  All other experimental settings remain the same.
    """

    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
        "parallel_tool_calls": True,
        "model_info": {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    }
    if model.casefold().startswith("ministral"):
        kwargs["add_name_prefixes"] = True
        kwargs["include_name_in_message"] = False
    return kwargs


def _build_model_client(
    *,
    thinking_mode: str | None,
    reasoning_effort: str | None,
    **kwargs: Any,
) -> ChatCompletionClient:
    """Build the recorded provider client without changing legacy defaults."""

    base_url = str(kwargs["base_url"]).rstrip("/").casefold()
    is_deepseek = base_url in {
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1",
    }
    is_qwen = str(kwargs["model"]).casefold().startswith("qwen")
    if is_qwen:
        if reasoning_effort is not None:
            raise ValueError(
                "Qwen thinking is controlled by a boolean mode and does not "
                "accept reasoning_effort."
            )
        if thinking_mode not in {None, "disabled", "enabled"}:
            raise ValueError("Qwen thinking_mode must be disabled or enabled.")
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": thinking_mode == "enabled"
            }
        }
        return OpenAIChatCompletionClient(**kwargs)

    if not is_deepseek:
        if thinking_mode is not None or reasoning_effort is not None:
            raise ValueError(
                "thinking_mode/reasoning_effort are supported only for Qwen "
                "or the direct DeepSeek API."
            )
        return OpenAIChatCompletionClient(**kwargs)

    if thinking_mode not in {"disabled", "enabled"}:
        raise ValueError(
            "Direct DeepSeek runs must explicitly set thinking_mode to "
            "'disabled' or 'enabled'."
        )
    if thinking_mode == "disabled" and reasoning_effort is not None:
        raise ValueError(
            "reasoning_effort must be omitted when thinking_mode is disabled."
        )
    if thinking_mode == "enabled" and reasoning_effort not in {
        "low",
        "high",
        "max",
    }:
        raise ValueError(
            "Thinking-enabled DeepSeek runs must record reasoning_effort as "
            "low, high, or max."
        )
    kwargs["extra_body"] = {"thinking": {"type": thinking_mode}}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    return DeepSeekChatCompletionClient(
        thinking_mode=thinking_mode,
        **kwargs,
    )


def _attack_succeeded_during_episode(
    *,
    observed_effect: bool,
    final_effect: bool,
) -> bool:
    """只要episode中曾验证目标状态变化，就保留攻击成功结果。

    DAG等拓扑可能沿多条路线重复执行同一个目标动作。后续重复调用可能
    让“最终状态恰好变化一次”的检查失败，但不能抹掉较早已经发生的
    危险效果。
    """

    return observed_effect or final_effect


def _reached_agents(
    recorder: RequestResponseRecorder,
) -> tuple[str, ...]:
    reached: list[str] = []
    seen: set[str] = set()
    for request in recorder.requests.values():
        if request.receiver_id not in seen:
            seen.add(request.receiver_id)
            reached.append(request.receiver_id)
    return tuple(reached)


def _usage_from_traces(
    traces: list[NormalAgentTrace],
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for trace in traces:
        for event in trace.autogen_events:
            usage = event.get("models_usage")
            if not isinstance(usage, Mapping):
                continue
            for source, target in (
                ("prompt_tokens", "prompt_tokens"),
                ("completion_tokens", "completion_tokens"),
            ):
                value = usage.get(source)
                if isinstance(value, int):
                    totals[target] = totals.get(target, 0) + value
    totals["total_tokens"] = (
        totals.get("prompt_tokens", 0)
        + totals.get("completion_tokens", 0)
    )
    return totals


def _usage_from_model_client(
    model_client: ChatCompletionClient,
) -> dict[str, int]:
    usage_accessor = getattr(model_client, "total_usage", None)
    raw_usage = (
        usage_accessor() if callable(usage_accessor) else usage_accessor
    )
    if raw_usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    if isinstance(raw_usage, Mapping):
        prompt_tokens = raw_usage.get("prompt_tokens", 0)
        completion_tokens = raw_usage.get("completion_tokens", 0)
    else:
        prompt_tokens = getattr(raw_usage, "prompt_tokens", 0)
        completion_tokens = getattr(raw_usage, "completion_tokens", 0)
    prompt = int(prompt_tokens) if isinstance(prompt_tokens, int) else 0
    completion = (
        int(completion_tokens)
        if isinstance(completion_tokens, int)
        else 0
    )
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    provider_usage = getattr(model_client, "provider_usage", None)
    if callable(provider_usage):
        provider_usage = provider_usage()
    if isinstance(provider_usage, Mapping):
        for key in (
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
            "reasoning_tokens",
        ):
            value = provider_usage.get(key)
            if isinstance(value, int):
                usage[key] = value
    return usage


def _usage_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    prompt = max(
        int(after.get("prompt_tokens", 0))
        - int(before.get("prompt_tokens", 0)),
        0,
    )
    completion = max(
        int(after.get("completion_tokens", 0))
        - int(before.get("completion_tokens", 0)),
        0,
    )
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    for key in (
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "reasoning_tokens",
    ):
        if key not in before and key not in after:
            continue
        usage[key] = max(
            int(after.get(key, 0)) - int(before.get(key, 0)),
            0,
        )
    return usage


def _merge_usage(
    client_usage: Mapping[str, int],
    trace_usage: Mapping[str, int],
) -> dict[str, int]:
    prompt = max(
        int(client_usage.get("prompt_tokens", 0)),
        int(trace_usage.get("prompt_tokens", 0)),
    )
    completion = max(
        int(client_usage.get("completion_tokens", 0)),
        int(trace_usage.get("completion_tokens", 0)),
    )
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    for key in (
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "reasoning_tokens",
    ):
        if key in client_usage:
            usage[key] = int(client_usage.get(key, 0))
    return usage
