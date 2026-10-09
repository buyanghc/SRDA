"""使用 AutoGen Core 承载 GatePath 的 Agent 间定向通信。

这一层只替换“消息怎样从一个 Agent 送到另一个 Agent”的基础设施，
不决定攻击策略，也不自动传播目标：

1. ``TeamWorld`` 仍是后台联络图和 capability 分布的唯一真相；
2. ``DelegationRequest`` 仍是 GatePath 已冻结的委托消息格式；
3. 每个接收 Agent 的具体行为由外部 handler 注入；
4. AutoGen Core 负责 Agent 身份、消息投递、请求和回复；
5. 本模块在发送前检查局部联络边，不能越过 ``local_contacts``。

攻击 LLM 不会直接拿到 ``AutoGenTeamRuntime``。它只能通过上层运行器提交
结构化决定；上层运行器使用当前真实 Agent 身份构造 DelegationRequest，
因此模型不能通过自己填写 ``sender`` 来冒充其他 Agent。
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import TypeAlias

from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)

from .protocol import ActionResult, DelegationRequest
from .world import TeamWorld


# handler 可以是同步函数，也可以是异步函数。第一阶段可注入确定性 worker，
# 后续可以在不修改通信层的情况下替换为 Qwen/其他 LLM worker。
AgentRequestHandler: TypeAlias = Callable[
    [str, DelegationRequest],
    ActionResult | Awaitable[ActionResult],
]


class _AutoGenNode(RoutedAgent):
    """AutoGen 中的一个内部 Agent 节点。

    ``logical_agent_id`` 是论文和 GatePath 配置使用的 Agent 名字，例如
    ``Agent_A``。AutoGen 自己还会分配一个运行时 AgentId；二者的映射只在
    后台保存，不进入攻击者 observation。
    """

    def __init__(
        self,
        *,
        logical_agent_id: str,
        runtime_ids: Mapping[str, AgentId],
        handler: AgentRequestHandler,
    ) -> None:
        super().__init__(description=f"GatePath node {logical_agent_id}")
        self._logical_agent_id = logical_agent_id
        self._runtime_ids = dict(runtime_ids)
        self._handler = handler

    @message_handler
    async def on_delegation_request(
        self,
        message: DelegationRequest,
        context: MessageContext,
    ) -> ActionResult:
        """校验消息身份，然后把请求交给该 Agent 的行为 handler。"""

        if message.recipient != self._logical_agent_id:
            raise ValueError(
                "DelegationRequest.recipient 与实际接收 Agent 不一致。"
            )

        expected_sender = self._runtime_ids.get(message.sender)
        if expected_sender is None:
            raise ValueError(f"未知的发送 Agent：{message.sender!r}")
        if context.sender != expected_sender:
            raise PermissionError(
                "DelegationRequest.sender 与 AutoGen 运行时身份不一致。"
            )

        result = self._handler(self._logical_agent_id, message)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, ActionResult):
            raise TypeError("Agent handler 必须返回 ActionResult。")
        return result


class AutoGenTeamRuntime:
    """把一个 ``TeamWorld`` 注册成可定向通信的 AutoGen Agent Team。

    该类不自动选择联系人，也不自动调用下游 Agent。只有上层攻击/worker
    决策明确产生 ``DelegationRequest`` 时，才发送一条消息。
    """

    def __init__(
        self,
        world: TeamWorld,
        handlers: Mapping[str, AgentRequestHandler],
    ) -> None:
        self._world = world
        self._handlers = dict(handlers)
        self._runtime = SingleThreadedAgentRuntime()
        self._runtime_ids = {
            logical_id: AgentId(f"gatepath_node_{index:04d}", "default")
            for index, logical_id in enumerate(world.agent_ids)
        }
        self._registered = False
        self._started = False
        self._validate_handlers()

    def _validate_handlers(self) -> None:
        expected = set(self._world.agent_ids)
        provided = set(self._handlers)
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        if missing or extra:
            raise ValueError(
                "handlers 必须与 TeamWorld Agent 一一对应；"
                f"missing={missing}, extra={extra}"
            )

    async def start(self) -> None:
        """注册全部 Agent，并启动本地 AutoGen 事件运行时。"""

        if self._started:
            raise RuntimeError("AutoGenTeamRuntime 已经启动。")

        if not self._registered:
            for logical_id in self._world.agent_ids:
                runtime_id = self._runtime_ids[logical_id]
                handler = self._handlers[logical_id]

                # 默认参数把本轮循环值冻结进 factory，避免所有 factory
                # 最后都引用循环中的同一个 Agent。
                await _AutoGenNode.register(
                    self._runtime,
                    runtime_id.type,
                    lambda logical_id=logical_id, handler=handler: _AutoGenNode(
                        logical_agent_id=logical_id,
                        runtime_ids=self._runtime_ids,
                        handler=handler,
                    ),
                )
            self._registered = True

        self._runtime.start()
        self._started = True

    async def stop(self) -> None:
        """等待已发送消息处理完成，然后关闭运行时。"""

        if not self._started:
            return
        await self._runtime.stop_when_idle()
        self._started = False

    async def send_delegation(
        self,
        request: DelegationRequest,
    ) -> ActionResult:
        """沿一条真实局部联络边发送请求，并等待接收 Agent 回复。

        此处检查的是通信边，不是工具 capability。接收 Agent 是否能执行
        目标动作，由它自己的 handler 和 GatePath 静态工具分配层决定。
        """

        if not self._started:
            raise RuntimeError("必须先启动 AutoGenTeamRuntime。")
        if not self._world.has_agent(request.sender):
            raise ValueError(f"未知发送 Agent：{request.sender!r}")
        if not self._world.has_agent(request.recipient):
            raise ValueError(f"未知接收 Agent：{request.recipient!r}")
        if request.recipient not in self._world.contacts_for(request.sender):
            raise PermissionError(
                f"{request.sender!r} 不能直接联系 {request.recipient!r}。"
            )

        result = await self._runtime.send_message(
            request,
            self._runtime_ids[request.recipient],
            sender=self._runtime_ids[request.sender],
            message_id=request.message_id,
        )
        if not isinstance(result, ActionResult):
            raise TypeError("AutoGen 接收 Agent 必须返回 ActionResult。")
        return result

    async def __aenter__(self) -> "AutoGenTeamRuntime":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        await self.stop()
