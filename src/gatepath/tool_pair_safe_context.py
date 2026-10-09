"""Token-limited context that never exposes orphaned tool messages."""

from __future__ import annotations

from autogen_core import FunctionCall
from autogen_core.model_context import TokenLimitedChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResultMessage,
    LLMMessage,
)


class ToolPairSafeTokenLimitedChatCompletionContext(
    TokenLimitedChatCompletionContext
):
    """Preserve OpenAI's tool-call/result history invariant after trimming.

    AutoGen 0.7.5's experimental token-limited context removes individual
    messages from the middle of the history.  It can therefore retain an
    ``AssistantMessage`` containing a function call while removing the
    adjacent ``FunctionExecutionResultMessage``.  OpenAI-compatible providers
    reject that history with ``No tool output found for function call``.

    The parent class still chooses the token-limited view.  This subclass only
    removes incomplete tool exchanges from that view; complete exchanges and
    all ordinary messages remain unchanged.
    """

    async def get_messages(self) -> list[LLMMessage]:
        messages = await super().get_messages()
        return remove_orphaned_tool_exchanges(messages)


def remove_orphaned_tool_exchanges(
    messages: list[LLMMessage],
) -> list[LLMMessage]:
    cleaned: list[LLMMessage] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if _is_tool_call_message(message):
            if index + 1 >= len(messages):
                index += 1
                continue
            result_message = messages[index + 1]
            if not isinstance(
                result_message,
                FunctionExecutionResultMessage,
            ):
                index += 1
                continue
            call_ids = {call.id for call in message.content}
            result_ids = {
                result.call_id for result in result_message.content
            }
            if call_ids == result_ids:
                cleaned.extend((message, result_message))
            index += 2
            continue
        if isinstance(message, FunctionExecutionResultMessage):
            index += 1
            continue
        cleaned.append(message)
        index += 1
    return cleaned


def _is_tool_call_message(message: LLMMessage) -> bool:
    return (
        isinstance(message, AssistantMessage)
        and isinstance(message.content, list)
        and bool(message.content)
        and all(isinstance(item, FunctionCall) for item in message.content)
    )
