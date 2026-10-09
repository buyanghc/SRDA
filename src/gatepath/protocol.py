"""GatePath 的五类信息流协议。

本模块把“Agent 能看到的局部信息”和“评测器掌握的全局真相”明确分开。
这是 GatePath 最重要的安全边界之一：

1. ``AgentObservation``：运行器交给当前 Agent/LLM 的局部观察；
2. ``AgentDecision``：LLM 返回给运行器的下一步决定；
3. ``DelegationRequest``：多 Agent 阶段使用的委托消息；
4. ``ActionResult``：工具、反馈披露层或下游 Agent 返回的可见结果；
5. ``AuditEvent``：只供后台评测器保存的完整事件。

``AuditEvent`` 中可能包含真实发起者和完整路径，因此绝不能被拼进
``AgentObservation``。路径编号（例如 ``route-0000``）只是一个不透明句柄，
可以用于关联请求与回复，但它本身不应编码真实 Agent 名字。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .models import FeedbackMessage


PROTOCOL_SCHEMA_VERSION = "0.2"


class DecisionKind(str, Enum):
    """LLM 在每轮只能从这三个动作中选择一个。"""

    ATTEMPT = "ATTEMPT"
    DELEGATE = "DELEGATE"
    STOP = "STOP"


class ResultStatus(str, Enum):
    """请求方能够观察到的结果类型。

    F0 的动作失败不会生成 ``ActionResult``；F1/F2 的动作失败统一是
    ``FAILURE``，两者只在 ``FeedbackMessage.reason_code`` 上不同。
    ``UNAVAILABLE`` 专用于联系人/路由接口不可用，不属于 F0/F1/F2 变量。
    """

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class AgentObservation:
    """当前 Agent/LLM 能看见的全部局部信息。

    ``remaining_steps`` 故意不存在。实验预算由运行器在后台管理，避免向
    Agent 泄露真实系统中未必存在的额外信息。
    """

    agent_id: str
    route_id: str
    objective: Mapping[str, Any]
    available_actions: tuple[str, ...] = ()
    reachable_agents: tuple[str, ...] = ()
    last_result: ActionResult | None = None

    def __post_init__(self) -> None:
        _require_text(self.agent_id, "agent_id")
        _require_text(self.route_id, "route_id")
        if not isinstance(self.objective, Mapping):
            raise ValueError("objective 必须是映射。")
        if len(set(self.reachable_agents)) != len(self.reachable_agents):
            raise ValueError("reachable_agents 不能包含重复 Agent。")
        for agent_id in self.reachable_agents:
            _require_text(agent_id, "reachable_agents 中的 agent_id")

    def as_dict(self) -> dict[str, Any]:
        """转换成实际发送给 LLM 的 JSON 对象。"""

        return {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "type": "observation",
            "agent_id": self.agent_id,
            "route_id": self.route_id,
            "objective": dict(self.objective),
            "local_view": {
                # available_actions 表示 Agent 可以发起请求的接口，不等于
                # 当前 Agent 静态拥有的真实本地工具集合。
                "available_actions": list(self.available_actions),
                "reachable_agents": list(self.reachable_agents),
            },
            "last_result": (
                None if self.last_result is None else self.last_result.as_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class AgentDecision:
    """LLM 返回的严格结构化决定。"""

    decision: DecisionKind
    recipient: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.decision is DecisionKind.DELEGATE:
            _require_text(self.recipient, "DELEGATE.recipient")
        elif self.recipient is not None:
            raise ValueError("只有 DELEGATE 决定可以包含 recipient。")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError("reason 必须是字符串或 null。")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AgentDecision":
        """校验并解析模型输出，拒绝自由文本和未知动作。"""

        if not isinstance(raw, Mapping):
            raise ValueError("模型输出必须是 JSON 对象。")
        raw_decision = raw.get("decision")
        try:
            decision = DecisionKind(str(raw_decision))
        except ValueError as exc:
            raise ValueError(
                "decision 必须是 ATTEMPT、DELEGATE 或 STOP。"
            ) from exc

        recipient = raw.get("recipient")
        reason = raw.get("reason")
        if recipient is not None and not isinstance(recipient, str):
            raise ValueError("recipient 必须是字符串或 null。")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("reason 必须是字符串或 null。")
        return cls(decision=decision, recipient=recipient, reason=reason)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "type": "decision",
            "decision": self.decision.value,
        }
        if self.recipient is not None:
            payload["recipient"] = self.recipient
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True, slots=True)
class DelegationRequest:
    """一个 Agent 发给直接联系人的委托请求。

    这里不包含 ``true_origin`` 或完整路径。接收方只能看到当前发送者、
    当前目标以及用于关联回复的不透明编号。
    """

    request_id: str
    route_id: str
    message_id: str
    sender: str
    recipient: str
    objective: Mapping[str, Any]
    parent_message_id: str | None = None

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.request_id, "request_id"),
            (self.route_id, "route_id"),
            (self.message_id, "message_id"),
            (self.sender, "sender"),
            (self.recipient, "recipient"),
        ):
            _require_text(value, field_name)
        if not isinstance(self.objective, Mapping):
            raise ValueError("objective 必须是映射。")
        if self.parent_message_id is not None:
            _require_text(self.parent_message_id, "parent_message_id")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "type": "delegation_request",
            "request_id": self.request_id,
            "route_id": self.route_id,
            "message_id": self.message_id,
            "parent_message_id": self.parent_message_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "objective": dict(self.objective),
        }


@dataclass(frozen=True, slots=True)
class ActionResult:
    """返回给请求方的可见结果。

    ``tool_dispatched`` 和 ``effect_verified`` 不放在这里，因为它们属于
    后台评测真相。Agent 只能看到系统实际披露的 status 和 feedback。
    F0 的失败不构造本对象，因此不会通过 ``last_result`` 泄露失败。
    """

    request_id: str
    route_id: str
    status: ResultStatus
    in_reply_to: str | None = None
    feedback: FeedbackMessage | None = None

    def __post_init__(self) -> None:
        _require_text(self.request_id, "request_id")
        _require_text(self.route_id, "route_id")
        if self.in_reply_to is not None:
            _require_text(self.in_reply_to, "in_reply_to")
        if (
            self.feedback is not None
            and self.status in {ResultStatus.SUCCESS, ResultStatus.FAILURE}
            and self.feedback.outcome != self.status.value
        ):
            raise ValueError("ActionResult status 与 feedback outcome 不一致。")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "type": "action_result",
            "request_id": self.request_id,
            "route_id": self.route_id,
            "in_reply_to": self.in_reply_to,
            "status": self.status.value,
            "feedback": (
                None if self.feedback is None else self.feedback.as_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """仅供后台评测器使用的一条全局事件。

    ``true_origin``、``path_prefix``、工具到达和状态验证结果均不得进入
    Agent 可见的 observation。``metadata`` 用于保存非核心调试信息，
    例如模型名称或解析重试次数。
    """

    episode_id: str
    event_id: str
    event_type: str
    request_id: str
    route_id: str
    true_origin: str
    current_agent: str
    hop_index: int
    path_prefix: tuple[str, ...]
    message_id: str | None = None
    parent_message_id: str | None = None
    current_sender: str | None = None
    receiver: str | None = None
    decision: AgentDecision | None = None
    result: ActionResult | None = None
    tool_dispatched: bool | None = None
    effect_verified: bool | None = None
    infrastructure_error: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.episode_id, "episode_id"),
            (self.event_id, "event_id"),
            (self.event_type, "event_type"),
            (self.request_id, "request_id"),
            (self.route_id, "route_id"),
            (self.true_origin, "true_origin"),
            (self.current_agent, "current_agent"),
        ):
            _require_text(value, field_name)
        if self.hop_index < 0:
            raise ValueError("hop_index 不能为负数。")
        if not self.path_prefix:
            raise ValueError("path_prefix 不能为空。")
        if self.path_prefix[-1] != self.current_agent:
            raise ValueError("path_prefix 末尾必须是 current_agent。")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "type": "audit_event",
            "episode_id": self.episode_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "request_id": self.request_id,
            "route_id": self.route_id,
            "message_id": self.message_id,
            "parent_message_id": self.parent_message_id,
            "true_origin": self.true_origin,
            "current_agent": self.current_agent,
            "current_sender": self.current_sender,
            "receiver": self.receiver,
            "hop_index": self.hop_index,
            "path_prefix": list(self.path_prefix),
            "decision": (
                None if self.decision is None else self.decision.as_dict()
            ),
            "result": None if self.result is None else self.result.as_dict(),
            "tool_dispatched": self.tool_dispatched,
            "effect_verified": self.effect_verified,
            "infrastructure_error": self.infrastructure_error,
            "metadata": dict(self.metadata),
        }
def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串。")
