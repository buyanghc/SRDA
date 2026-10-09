"""Compound semantic episodes with a normal workflow and optional attack."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping

from .agentdojo_adapter import AgentDojoExecutor
from .agentdojo_verifier import AgentDojoTargetVerifier
from .request_response_protocol import (
    EntryAgentMode,
    HandoffContinuation,
    RequestResponseBudget,
    RequestResponsePromptBundle,
    ResponseVisibility,
)
from .request_response_runner import (
    RequestResponseEpisodeReport,
    run_request_response_episode_async,
)
from .workflow_graphs import WorkflowGraphSpec
from .workflow_scenarios import (
    WorkflowCasePair,
    build_workflow_request_content,
    prerequisite_actions_executed,
)


MIXED_WORKFLOW_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class MixedWorkflowReport:
    episode_id: str
    condition: str
    regime: str
    normal_case: WorkflowCasePair
    attack_case: WorkflowCasePair
    graph: WorkflowGraphSpec
    benign_workflow_success: bool
    attack_success: bool | None
    joint_success: bool | None
    benign_effect_persists_at_end: bool
    attack_effect_persists_at_end: bool | None
    prerequisite_actions_executed: Mapping[str, bool]
    normal_request_content: str
    attack_request_content: str | None
    normal_report: RequestResponseEpisodeReport
    attack_report: RequestResponseEpisodeReport | None

    def as_dict(self) -> dict[str, Any]:
        reports = (self.normal_report,) + (
            () if self.attack_report is None else (self.attack_report,)
        )
        combined_usage: dict[str, int] = {}
        for report in reports:
            for key, value in report.usage.items():
                combined_usage[key] = combined_usage.get(key, 0) + int(value)
        return {
            "mixed_workflow_schema_version": MIXED_WORKFLOW_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "condition": self.condition,
            "regime": self.regime,
            "execution_mode": (
                "normal_request_only"
                if self.attack_report is None
                else "concurrent_requests_shared_environment"
            ),
            "shared_environment": self.attack_report is not None,
            "normal_case": self.normal_case.as_dict(),
            "attack_case": self.attack_case.as_dict(),
            "workflow_graph": self.graph.as_dict(),
            "benign_workflow_success": self.benign_workflow_success,
            "attack_success": self.attack_success,
            "joint_success": self.joint_success,
            "benign_effect_persists_at_end": (
                self.benign_effect_persists_at_end
            ),
            "attack_effect_persists_at_end": (
                self.attack_effect_persists_at_end
            ),
            "prerequisite_actions_executed": dict(
                self.prerequisite_actions_executed
            ),
            "normal_request_content": self.normal_request_content,
            "attack_request_content": self.attack_request_content,
            "combined_messages_sent": sum(
                report.messages_sent for report in reports
            ),
            "combined_usage": combined_usage,
            "normal_request_report": self.normal_report.as_dict(),
            "attack_request_report": (
                None
                if self.attack_report is None
                else self.attack_report.as_dict()
            ),
        }


async def run_mixed_workflow_episode_async(
    *,
    episode_id: str,
    condition: str,
    regime: str,
    normal_case: WorkflowCasePair,
    attack_case: WorkflowCasePair,
    graph: WorkflowGraphSpec,
    legitimate_prompts: RequestResponsePromptBundle,
    attack_prompts: RequestResponsePromptBundle,
    budget: RequestResponseBudget,
    response_visibility: ResponseVisibility,
    handoff_continuation: HandoffContinuation,
    model: str,
    base_url: str,
    temperature: float,
) -> MixedWorkflowReport:
    if condition not in {"normal_only", "normal_plus_attack"}:
        raise ValueError("Unsupported mixed-workflow condition.")
    if normal_case.suite_name != attack_case.suite_name:
        raise ValueError("Normal and attack cases must use the same suite.")
    if graph.suite_name != normal_case.suite_name:
        raise ValueError("Workflow graph and cases must use the same suite.")
    normal_instance = normal_case.target_instance()
    attack_instance = attack_case.target_instance()
    if normal_instance.action != graph.target_action:
        raise ValueError("Normal case action does not match the workflow graph.")
    if attack_instance.action != graph.target_action:
        raise ValueError("Attack case action does not match the workflow graph.")
    if dict(normal_instance.parameters) == dict(attack_instance.parameters):
        raise ValueError("Normal and attack target parameters must differ.")

    executor = AgentDojoExecutor(graph.suite_name)
    executor.reset()
    verifier = AgentDojoTargetVerifier()
    normal_before = verifier.capture_if_formal(
        normal_instance.suite_name,
        normal_instance.action,
        executor.backend_environment,
    )
    attack_before = verifier.capture_if_formal(
        attack_instance.suite_name,
        attack_instance.action,
        executor.backend_environment,
    )
    if normal_before is None or attack_before is None:
        raise RuntimeError("Mixed workflow requires two formal target actions.")

    normal_topology = graph.topology(task_kind="legitimate")
    attack_topology = graph.topology(task_kind="attack")
    normal_request_content = build_workflow_request_content(
        case=normal_case,
        task_kind="legitimate",
        workflow_note=legitimate_prompts.workflow_note,
    )
    normal_call = run_request_response_episode_async(
        instance=normal_instance,
        topology=normal_topology,
        prompts=legitimate_prompts,
        condition=f"MIXED_{condition.upper()}_{regime.upper()}_BENIGN",
        budget=budget,
        episode_id=f"{episode_id}-BENIGN",
        executor=executor,
        model=model,
        base_url=base_url,
        temperature=temperature,
        entry_agent_mode=EntryAgentMode.LLM,
        response_visibility=response_visibility,
        handoff_continuation=handoff_continuation,
        assigned_team_override=graph.assigned_team(normal_topology),
        request_content_override=normal_request_content,
        reset_executor=False,
    )

    attack_request_content: str | None = None
    attack_report: RequestResponseEpisodeReport | None = None
    if condition == "normal_plus_attack":
        attack_request_content = build_workflow_request_content(
            case=attack_case,
            task_kind="attack",
            workflow_note=attack_prompts.workflow_note,
        )
        attack_call = run_request_response_episode_async(
            instance=attack_instance,
            topology=attack_topology,
            prompts=attack_prompts,
            condition=f"MIXED_{condition.upper()}_{regime.upper()}_ATTACK",
            budget=budget,
            episode_id=f"{episode_id}-ATTACK",
            executor=executor,
            model=model,
            base_url=base_url,
            temperature=temperature,
            entry_agent_mode=EntryAgentMode.DETERMINISTIC,
            response_visibility=response_visibility,
            handoff_continuation=handoff_continuation,
            assigned_team_override=graph.assigned_team(attack_topology),
            request_content_override=attack_request_content,
            reset_executor=False,
        )
        normal_report, attack_report = await asyncio.gather(
            normal_call,
            attack_call,
        )
    else:
        normal_report = await normal_call

    prerequisite_status = prerequisite_actions_executed(
        required_actions=normal_case.legitimate_prerequisite_actions,
        audit_events=normal_report.audit_events,
    )
    benign_success = normal_report.target_effect_verified and all(
        prerequisite_status.values()
    )
    attack_success = (
        None if attack_report is None else attack_report.target_effect_verified
    )
    final_normal = verifier.verify_if_formal(
        normal_instance.suite_name,
        normal_instance.action,
        normal_instance.parameters,
        normal_before,
        executor.backend_environment,
    )
    final_attack = verifier.verify_if_formal(
        attack_instance.suite_name,
        attack_instance.action,
        attack_instance.parameters,
        attack_before,
        executor.backend_environment,
    )
    if final_normal is None or final_attack is None:
        raise RuntimeError("Mixed workflow final-state verification failed.")

    return MixedWorkflowReport(
        episode_id=episode_id,
        condition=condition,
        regime=regime,
        normal_case=normal_case,
        attack_case=attack_case,
        graph=graph,
        benign_workflow_success=benign_success,
        attack_success=attack_success,
        joint_success=(
            None
            if attack_success is None
            else benign_success and attack_success
        ),
        benign_effect_persists_at_end=final_normal.succeeded,
        attack_effect_persists_at_end=(
            None if attack_report is None else final_attack.succeeded
        ),
        prerequisite_actions_executed=prerequisite_status,
        normal_request_content=normal_request_content,
        attack_request_content=attack_request_content,
        normal_report=normal_report,
        attack_report=attack_report,
    )


def run_mixed_workflow_episode(**kwargs: Any) -> MixedWorkflowReport:
    return asyncio.run(run_mixed_workflow_episode_async(**kwargs))
