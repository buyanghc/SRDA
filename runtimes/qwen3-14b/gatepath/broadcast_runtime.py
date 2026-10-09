"""使用 AutoGen Core Direct Messaging 传输第一阶段 WORK_REQUEST。"""

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

from .broadcast_protocol import (
    BroadcastAgentRun,
    BroadcastWorkRequest,
)
from .world import TeamWorld


BroadcastRequestHandler: TypeAlias = Callable[
    [str, BroadcastWorkRequest],
    BroadcastAgentRun | Awaitable[BroadcastAgentRun],
]


class _AutoGenBroadcastNode(RoutedAgent):
    def __init__(
        self,
        *,
        logical_agent_id: str,
        runtime_ids: Mapping[str, AgentId],
        handler: BroadcastRequestHandler,
    ) -> None:
        super().__init__(
            description=f"GatePath team node {logical_agent_id}"
        )
        self._logical_agent_id = logical_agent_id
        self._runtime_ids = dict(runtime_ids)
        self._handler = handler

    @message_handler
    async def on_work_request(
        self,
        message: BroadcastWorkRequest,
        context: MessageContext,
    ) -> BroadcastAgentRun:
        if message.receiver_id != self._logical_agent_id:
            raise ValueError(
                "WORK_REQUEST.receiver_id 与实际接收 Agent 不一致。"
            )
        expected_sender = self._runtime_ids.get(message.sender_id)
        if expected_sender is None:
            raise ValueError(f"未知发送 Agent：{message.sender_id!r}")
        if context.sender != expected_sender:
            raise PermissionError(
                "WORK_REQUEST.sender_id 与 AutoGen 运行时身份不一致。"
            )

        result = self._handler(self._logical_agent_id, message)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, BroadcastAgentRun):
            raise TypeError(
                "Broadcast Agent handler 必须返回 BroadcastAgentRun。"
            )
        return result


class AutoGenBroadcastRuntime:
    """真实 AutoGen Agent 身份和局部有向边上的 Direct-message 投递层。

    ``send_request`` 返回值只交给可信运行器，用于执行接收 Agent 的决定。
    第一阶段不会把该下游结果加入 sender 的 LLM 输入，因此仍满足
    No-downstream-disclosure；这不改变 AutoGen Core 的一对一投递机制。
    """

    def __init__(
        self,
        world: TeamWorld,
        handlers: Mapping[str, BroadcastRequestHandler],
    ) -> None:
        self._world = world
        self._handlers = dict(handlers)
        self._runtime = SingleThreadedAgentRuntime()
        self._runtime_ids = {
            logical_id: AgentId(
                f"gatepath_broadcast_node_{index:04d}",
                "default",
            )
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
        if self._started:
            raise RuntimeError("AutoGenBroadcastRuntime 已经启动。")
        if not self._registered:
            for logical_id in self._world.agent_ids:
                runtime_id = self._runtime_ids[logical_id]
                handler = self._handlers[logical_id]
                await _AutoGenBroadcastNode.register(
                    self._runtime,
                    runtime_id.type,
                    lambda logical_id=logical_id, handler=handler: (
                        _AutoGenBroadcastNode(
                            logical_agent_id=logical_id,
                            runtime_ids=self._runtime_ids,
                            handler=handler,
                        )
                    ),
                )
            self._registered = True
        self._runtime.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        await self._runtime.stop_when_idle()
        self._started = False

    async def send_request(
        self,
        request: BroadcastWorkRequest,
    ) -> BroadcastAgentRun:
        """沿当前 sender 的一条真实局部边投递请求。"""

        if not self._started:
            raise RuntimeError("必须先启动 AutoGenBroadcastRuntime。")
        if not self._world.has_agent(request.sender_id):
            raise ValueError(f"未知发送 Agent：{request.sender_id!r}")
        if not self._world.has_agent(request.receiver_id):
            raise ValueError(f"未知接收 Agent：{request.receiver_id!r}")
        if request.receiver_id not in self._world.contacts_for(
            request.sender_id
        ):
            raise PermissionError(
                f"{request.sender_id!r} 不能直接联系 "
                f"{request.receiver_id!r}。"
            )

        result = await self._runtime.send_message(
            request,
            self._runtime_ids[request.receiver_id],
            sender=self._runtime_ids[request.sender_id],
            message_id=request.request_id,
        )
        if not isinstance(result, BroadcastAgentRun):
            raise TypeError(
                "AutoGen team Agent 必须返回 BroadcastAgentRun。"
            )
        return result

    async def __aenter__(self) -> "AutoGenBroadcastRuntime":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        await self.stop()
