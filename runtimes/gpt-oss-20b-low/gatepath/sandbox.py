"""GatePath 与外部场景/沙箱之间的通用接口。

这里不内置任何具体动作，也不提供假删除、假付款等预设场景。
正式场景由 AgentDojo 实现 ActionExecutor。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ExecutionResult:
    """沙箱执行一次 action 的分层结果。

    success 是正式目标效果是否成功；其余字段只用于区分路径/工具/环境故障。
    对非 AgentDojo 的测试执行器，新增字段都有安全默认值。
    """

    success: bool
    message: str
    tool_dispatched: bool = False
    effect_verified: bool | None = None
    infrastructure_error: bool = False


class ActionExecutor(Protocol):
    """GatePath 对外部沙箱的最小要求。

    任何后端只要实现 execute，就能接到 GatePath 环境里。
    """

    def execute(
        self,
        agent_id: str,
        action: str,
        parameters: Mapping[str, Any],
    ) -> ExecutionResult:
        ...
