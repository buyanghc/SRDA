"""GatePath 各模块共用的数据结构。

这里严格区分两类信息：
1. ``InternalResultCode`` 是实验后台掌握的真实失败原因；
2. ``FeedbackMessage`` 是反馈披露层允许 Agent/LLM 看到的消息。

两者不能混用。尤其在 F0 条件下，后台仍然知道请求为什么失败，但模型
不会收到任何失败消息。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FeedbackLevel(str, Enum):
    """一次失败后，反馈披露层对外暴露的信息等级。"""

    F0 = "F0"
    F1 = "F1"
    F2 = "F2"


class InternalResultCode(str, Enum):
    """只供实验后台使用的动作结果分类。

    这些代码用于评测和调试。它们是否会被 Agent 看到，完全由
    ``FeedbackDisclosureLayer`` 和当前 F0/F1/F2 条件决定。
    """

    ACTION_SUCCEEDED = "ACTION_SUCCEEDED"
    LOCAL_TOOL_UNAVAILABLE = "LOCAL_TOOL_UNAVAILABLE"
    ACTION_STATE_UNCHANGED = "ACTION_STATE_UNCHANGED"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"


@dataclass(frozen=True)
class FeedbackMessage:
    """反馈披露层实际返回给请求方的结构化消息。"""

    outcome: str
    reason_code: str | None

    def __post_init__(self) -> None:
        if self.outcome not in {"SUCCESS", "FAILURE"}:
            raise ValueError("feedback outcome 只能是 SUCCESS 或 FAILURE。")
        if self.outcome == "SUCCESS" and self.reason_code is not None:
            raise ValueError("SUCCESS feedback 不能包含失败原因码。")

    def as_dict(self) -> dict[str, str | None]:
        """转成可以直接写入 JSON 日志的格式。"""

        return {
            "outcome": self.outcome,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class RouteHandle:
    """攻击者能看到的匿名路径句柄。

    token 不包含真实 Agent 名字。hops 只表示已经委托了多少次，
    不会暴露完整联络图。
    """

    token: str
    hops: int

    def as_dict(self) -> dict[str, Any]:
        return {"token": self.token, "hops": self.hops}


@dataclass(frozen=True)
class AttemptResult:
    """Environment 返回给运行器的内部尝试结果。

    运行器可以读取 ``success`` 和 ``internal_result_code`` 来做后台评测，
    但只能把 ``feedback`` 交给 Agent/LLM。这个对象本身不是模型输入。
    """

    route: RouteHandle
    success: bool
    internal_result_code: InternalResultCode
    feedback: FeedbackMessage | None


@dataclass(frozen=True)
class DelegationObservation:
    """运行器请求当前 Agent 继续委托后获得的局部结果。

    new_routes 只包含匿名 token。环境不会在这里返回完整路径。
    """

    source_route: RouteHandle
    new_routes: tuple[RouteHandle, ...]
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_route": self.source_route.as_dict(),
            "new_routes": [route.as_dict() for route in self.new_routes],
            "status": self.status,
        }


@dataclass(frozen=True)
class InteractionEvent:
    """只供后台评测器使用的一条完整事件。

    hidden_path 和 feedback_return_path 包含真实 Agent 名字，因此绝不能
    传给攻击者。它们只用于计算真实路径、检查反馈是否按原路返回。
    """

    index: int
    kind: str
    route_token: str
    hidden_path: tuple[str, ...]
    feedback_return_path: tuple[str, ...]
    success: bool | None
    internal_result_code: InternalResultCode | None
    feedback: FeedbackMessage | None
    status: str | None
    tool_dispatched: bool | None = None
    effect_verified: bool | None = None
    infrastructure_error: bool = False
    execution_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "route_token": self.route_token,
            "hidden_path": list(self.hidden_path),
            "feedback_return_path": list(self.feedback_return_path),
            "success": self.success,
            "internal_result_code": (
                None
                if self.internal_result_code is None
                else self.internal_result_code.value
            ),
            "feedback": None if self.feedback is None else self.feedback.as_dict(),
            "status": self.status,
            "tool_dispatched": self.tool_dispatched,
            "effect_verified": self.effect_verified,
            "infrastructure_error": self.infrastructure_error,
            "execution_message": self.execution_message,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """一次 episode 的后台评测结果。"""

    scenario_name: str
    attack_success: bool
    forbidden_action_executed: bool
    total_interactions: int
    execution_attempts: int
    local_tool_unavailable_failures: int
    delegation_requests: int
    routes_discovered: int
    successful_path: tuple[str, ...] | None
    global_shortest_path: tuple[str, ...] | None
    shortest_allowed_path: tuple[str, ...] | None
    path_length: int | None
    path_excess_hops: int | None
    events: tuple[InteractionEvent, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "attack_success": self.attack_success,
            "forbidden_action_executed": self.forbidden_action_executed,
            "total_interactions": self.total_interactions,
            "execution_attempts": self.execution_attempts,
            "local_tool_unavailable_failures": (
                self.local_tool_unavailable_failures
            ),
            "delegation_requests": self.delegation_requests,
            "routes_discovered": self.routes_discovered,
            "successful_path": (
                None if self.successful_path is None else list(self.successful_path)
            ),
            "global_shortest_path": (
                None
                if self.global_shortest_path is None
                else list(self.global_shortest_path)
            ),
            "shortest_allowed_path": (
                None
                if self.shortest_allowed_path is None
                else list(self.shortest_allowed_path)
            ),
            "path_length": self.path_length,
            "path_excess_hops": self.path_excess_hops,
            "events": [event.as_dict() for event in self.events],
        }
