"""Token-limited context that never exposes orphaned tool messages."""

from __future__ import annotations

from typing import Any

from autogen_core import FunctionCall
from autogen_core.model_context import TokenLimitedChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResultMessage,
    LLMMessage,
    UserMessage,
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

    By default, the parent class still chooses the token-limited view and this
    subclass only removes incomplete tool exchanges.  Strict-role providers
    additionally retain the latest request and close historical tool turns
    before a later user message.
    """

    def __init__(
        self,
        *args: Any,
        strict_role_order: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._strict_role_order = strict_role_order

    async def get_messages(self) -> list[LLMMessage]:
        if not self._strict_role_order:
            messages = await super().get_messages()
            return remove_orphaned_tool_exchanges(messages)

        messages = _token_limited_messages_preserving_latest(self)
        messages = remove_orphaned_tool_exchanges(messages)
        return normalize_strict_tool_history(messages)


def _token_limited_messages_preserving_latest(
    context: ToolPairSafeTokenLimitedChatCompletionContext,
) -> list[LLMMessage]:
    """Apply AutoGen's token limit without deleting the latest message."""

    messages = list(context._messages)
    if context._token_limit is None:
        remaining = context._model_client.remaining_tokens(
            messages,
            tools=context._tool_schema,
        )
        while remaining < 0 and len(messages) > 1:
            messages.pop(_removal_index_preserving_latest(messages))
            remaining = context._model_client.remaining_tokens(
                messages,
                tools=context._tool_schema,
            )
    else:
        token_count = context._model_client.count_tokens(
            messages,
            tools=context._tool_schema,
        )
        while token_count > context._token_limit and len(messages) > 1:
            messages.pop(_removal_index_preserving_latest(messages))
            token_count = context._model_client.count_tokens(
                messages,
                tools=context._tool_schema,
            )
    if messages and isinstance(messages[0], FunctionExecutionResultMessage):
        messages = messages[1:]
    return messages


def _removal_index_preserving_latest(messages: list[LLMMessage]) -> int:
    return min(len(messages) - 2, len(messages) // 2)


def normalize_strict_tool_history(
    messages: list[LLMMessage],
) -> list[LLMMessage]:
    """Close completed historical tool turns for strict chat templates."""

    normalized = list(messages)
    while normalized and isinstance(normalized[-1], AssistantMessage):
        normalized.pop()

    closed: list[LLMMessage] = []
    for index, message in enumerate(normalized):
        closed.append(message)
        if not isinstance(message, FunctionExecutionResultMessage):
            continue
        if index + 1 >= len(normalized) or not isinstance(
            normalized[index + 1],
            UserMessage,
        ):
            continue
        source = _preceding_assistant_source(closed)
        summary = "\n".join(result.content for result in message.content)
        closed.append(
            AssistantMessage(
                content=summary or "Tool execution completed.",
                source=source,
            )
        )
    return closed


def _preceding_assistant_source(messages: list[LLMMessage]) -> str:
    for message in reversed(messages[:-1]):
        if isinstance(message, AssistantMessage):
            return message.source
    return "assistant"


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
