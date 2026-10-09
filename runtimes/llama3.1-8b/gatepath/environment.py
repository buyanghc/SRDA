"""GatePath 的通用执行环境。

这个环境从第一版就支持：
- 单 Agent 直接尝试；
- 任意有向联络图；
- 0 跳、1 跳或多跳委托；
- 匿名路径 token；
- 下游失败反馈沿原路径返回；
- 后台真实最短路径评测。

当前单 Agent 配置不会使用委托功能，但以后不需要重写这部分。
"""

from __future__ import annotations

from typing import Callable

from .evaluator import build_report
from .feedback import FeedbackDisclosureLayer
from .models import (
    AttemptResult,
    DelegationObservation,
    EvaluationReport,
    InternalResultCode,
    InteractionEvent,
    RouteHandle,
)
from .sandbox import ActionExecutor, ExecutionResult
from .world import Scenario


class GatePathSession:
    """可信运行器访问 Environment 的窄接口。

    Session 没有 world、graph、capabilities 或 shortest_path 属性，但它仍会
    返回后台 ``AttemptResult``，所以绝不能直接交给 Agent/LLM。模型只能看到
    运行器根据披露规则构造的 ``AgentObservation``。
    """

    __slots__ = ("__begin_fn", "__attempt_fn", "__delegate_fn")

    def __init__(
        self,
        begin_fn: Callable[[], RouteHandle],
        attempt_fn: Callable[[RouteHandle], AttemptResult],
        delegate_fn: Callable[[RouteHandle], DelegationObservation],
    ) -> None:
        self.__begin_fn = begin_fn
        self.__attempt_fn = attempt_fn
        self.__delegate_fn = delegate_fn

    def begin(self) -> RouteHandle:
        """获得指向起始 Agent 的匿名路线。"""

        return self.__begin_fn()

    def attempt(self, route: RouteHandle) -> AttemptResult:
        """让 route 当前指向的 Agent 尝试执行固定目标动作。

        返回值供可信运行器使用，不会直接交给 Agent/LLM。运行器只能向模型
        公开其中的 ``feedback``。
        """

        return self.__attempt_fn(route)

    def delegate(self, route: RouteHandle) -> DelegationObservation:
        """请求 route 当前指向的 Agent 暴露其本地匿名委托出口。

        这里只生成候选路线，不会自动选择，也不会自动执行下一步。
        将来由攻击者 LLM 决定使用哪一个 new_route。
        """

        return self.__delegate_fn(route)


class GatePathEnvironment:
    """掌握完整世界真相、静态工具分配和后台评测事件。"""

    def __init__(self, scenario: Scenario, executor: ActionExecutor) -> None:
        self._scenario = scenario
        self._executor = executor
        self._feedback_disclosure = FeedbackDisclosureLayer(
            scenario.feedback_level
        )
        self._routes: dict[str, tuple[str, ...]] = {}
        self._route_counter = 0
        self._events: list[InteractionEvent] = []
        self._forbidden_action_executed = False
        self._successful_path: tuple[str, ...] | None = None
        self._started = False

    def runner_session(self) -> GatePathSession:
        """创建可信运行器接口；该对象不能放进模型上下文。"""

        return GatePathSession(self._begin, self._attempt, self._delegate)

    def _begin(self) -> RouteHandle:
        if self._started:
            raise RuntimeError("一个 Environment 只能运行一个 episode。")
        self._started = True
        return self._new_route((self._scenario.start_agent,))

    def _attempt(self, route: RouteHandle) -> AttemptResult:
        path = self._resolve_route(route)
        self._check_budget()
        current_agent = path[-1]

        success = False
        tool_dispatched = False
        effect_verified: bool | None = False
        infrastructure_error = False
        execution_message: str | None = None
        if self._scenario.world.can_execute(
            current_agent,
            self._scenario.target_action,
        ):
            execution = self._executor.execute(
                current_agent,
                self._scenario.target_action,
                self._scenario.target_parameters,
            )
            success = execution.success
            tool_dispatched = execution.tool_dispatched
            effect_verified = execution.effect_verified
            infrastructure_error = execution.infrastructure_error
            execution_message = execution.message
            internal_result_code = _classify_execution_result(execution)
            if success:
                self._forbidden_action_executed = True
                self._successful_path = path
        else:
            # 当前 Agent 的静态工具集合中没有目标工具。这里是本地能力边界，
            # 不是另设一个会动态批准或拒绝请求的显式授权 gate。
            internal_result_code = InternalResultCode.LOCAL_TOOL_UNAVAILABLE

        feedback = self._feedback_disclosure.disclose(
            success=success,
            internal_result_code=internal_result_code,
        )

        # 下游反馈按照真实委托路径反向返回。
        # 例如隐藏路径 A→B→C，则后台记录 C→B→A。
        # Agent/LLM 最多只收到 feedback，不会收到这些真实 Agent 名字。
        feedback_return_path = tuple(reversed(path))
        self._events.append(
            InteractionEvent(
                index=len(self._events) + 1,
                kind="ATTEMPT",
                route_token=route.token,
                hidden_path=path,
                feedback_return_path=feedback_return_path,
                success=success,
                internal_result_code=internal_result_code,
                feedback=feedback,
                status=None,
                tool_dispatched=tool_dispatched,
                effect_verified=effect_verified,
                infrastructure_error=infrastructure_error,
                execution_message=execution_message,
            )
        )
        return AttemptResult(
            route=route,
            success=success,
            internal_result_code=internal_result_code,
            feedback=feedback,
        )

    def _delegate(self, route: RouteHandle) -> DelegationObservation:
        path = self._resolve_route(route)
        self._check_budget()
        current_hops = len(path) - 1

        if (
            self._scenario.max_hops is not None
            and current_hops >= self._scenario.max_hops
        ):
            status = "MAX_HOPS_REACHED"
            new_routes: tuple[RouteHandle, ...] = ()
        else:
            contacts = self._scenario.world.contacts_for(path[-1])
            new_routes = tuple(
                self._new_route(path + (contact,)) for contact in contacts
            )
            status = "ROUTES_AVAILABLE" if new_routes else "LOCAL_DEAD_END"

        self._events.append(
            InteractionEvent(
                index=len(self._events) + 1,
                kind="DELEGATE",
                route_token=route.token,
                hidden_path=path,
                feedback_return_path=(),
                success=None,
                internal_result_code=None,
                feedback=None,
                status=status,
            )
        )
        return DelegationObservation(
            source_route=route,
            new_routes=new_routes,
            status=status,
        )

    def _new_route(self, path: tuple[str, ...]) -> RouteHandle:
        token = f"route-{self._route_counter:04d}"
        self._route_counter += 1
        self._routes[token] = path
        return RouteHandle(token=token, hops=len(path) - 1)

    def _resolve_route(self, route: RouteHandle) -> tuple[str, ...]:
        try:
            path = self._routes[route.token]
        except KeyError as exc:
            raise ValueError("未知或伪造的 route token。") from exc
        if route.hops != len(path) - 1:
            raise ValueError("route 的 hops 与后台记录不一致。")
        return path

    def _check_budget(self) -> None:
        if len(self._events) >= self._scenario.interaction_budget:
            raise RuntimeError("INTERACTION_BUDGET_EXHAUSTED")

    def evaluator_report(self) -> EvaluationReport:
        """返回只供实验评测者查看的完整报告。"""

        return build_report(
            scenario=self._scenario,
            events=tuple(self._events),
            forbidden_action_executed=self._forbidden_action_executed,
            successful_path=self._successful_path,
            routes_discovered=len(self._routes),
        )


def _classify_execution_result(
    execution: ExecutionResult,
) -> InternalResultCode:
    """把沙箱结果归入后台原因码，不决定对外披露多少信息。"""

    if execution.success:
        return InternalResultCode.ACTION_SUCCEEDED
    if execution.infrastructure_error:
        return InternalResultCode.INFRASTRUCTURE_ERROR
    if execution.tool_dispatched and execution.effect_verified is False:
        return InternalResultCode.ACTION_STATE_UNCHANGED
    return InternalResultCode.TOOL_EXECUTION_FAILED
