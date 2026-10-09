"""DeepSeek-compatible AutoGen model client.

DeepSeek thinking-mode tool calls require every assistant
``reasoning_content`` value to be returned in later requests.  AutoGen 0.7.5
stores that value as ``AssistantMessage.thought`` but its generic OpenAI
serializer does not emit the provider-specific field.  This client restores
the field without changing AutoGen's public message objects and also retains
DeepSeek's cache/reasoning token counters for exact cost accounting.
"""

from __future__ import annotations

import asyncio
import json
from asyncio import Task
from typing import Any, Literal, Mapping, Sequence, Union, cast

from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import (
    AssistantMessage,
    CreateResult,
    LLMMessage,
    RequestUsage,
)
from autogen_core.tools import Tool, ToolSchema
from autogen_ext.models._utils.normalize_stop_reason import (
    normalize_stop_reason,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.openai._openai_client import normalize_name
from openai import NOT_GIVEN
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from pydantic import BaseModel


_PROVIDER_USAGE_KEYS = (
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "reasoning_tokens",
)


class DeepSeekChatCompletionClient(OpenAIChatCompletionClient):
    """OpenAI-compatible DeepSeek client with safe thinking-tool history."""

    def __init__(
        self,
        *,
        thinking_mode: Literal["disabled", "enabled"],
        **kwargs: Any,
    ) -> None:
        self._deepseek_thinking_mode = thinking_mode
        self._deepseek_provider_usage = {
            key: 0 for key in _PROVIDER_USAGE_KEYS
        }
        super().__init__(**kwargs)

    @property
    def provider_usage(self) -> Mapping[str, int]:
        """Return provider-specific cumulative counters without credentials."""

        return dict(self._deepseek_provider_usage)

    def _process_create_args(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema],
        tool_choice: Tool | Literal["auto", "required", "none"],
        json_output: bool | type[BaseModel] | None,
        extra_create_args: Mapping[str, Any],
    ) -> Any:
        params = super()._process_create_args(
            messages,
            tools,
            tool_choice,
            json_output,
            extra_create_args,
        )
        if self._deepseek_thinking_mode == "enabled":
            _restore_reasoning_content(messages, params.messages)
        return params

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = (),
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        """Create one completion while preserving DeepSeek reasoning history."""

        create_params = self._process_create_args(
            messages,
            tools,
            tool_choice,
            json_output,
            extra_create_args,
        )
        future: Union[Task[Any], Task[ChatCompletion]]
        if create_params.response_format is not None:
            future = asyncio.ensure_future(
                self._client.beta.chat.completions.parse(
                    messages=create_params.messages,
                    tools=(
                        create_params.tools
                        if create_params.tools
                        else NOT_GIVEN
                    ),
                    response_format=create_params.response_format,
                    **create_params.create_args,
                )
            )
        else:
            future = asyncio.ensure_future(
                self._client.chat.completions.create(
                    messages=create_params.messages,
                    stream=False,
                    tools=(
                        create_params.tools
                        if create_params.tools
                        else NOT_GIVEN
                    ),
                    **create_params.create_args,
                )
            )
        if cancellation_token is not None:
            cancellation_token.link_future(future)
        result = await future
        if create_params.response_format is not None:
            result = cast(ParsedChatCompletion[Any], result)

        usage = _request_usage(result.usage)
        self._total_usage = _add_request_usage(self._total_usage, usage)
        self._actual_usage = _add_request_usage(self._actual_usage, usage)
        _accumulate_provider_usage(
            self._deepseek_provider_usage,
            result.usage,
        )

        choice = result.choices[0]
        if choice.message.function_call is not None:
            raise ValueError(
                "function_call is deprecated and unsupported; use tool_calls."
            )
        thought = _reasoning_content(choice.message)
        if choice.message.tool_calls:
            calls: list[FunctionCall] = []
            for tool_call in choice.message.tool_calls:
                arguments = tool_call.function.arguments
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, sort_keys=True)
                calls.append(
                    FunctionCall(
                        id=tool_call.id,
                        name=normalize_name(tool_call.function.name),
                        arguments=arguments,
                    )
                )
            content: str | list[FunctionCall] = calls
            if thought is None and choice.message.content:
                thought = choice.message.content
            finish_reason = "tool_calls"
        else:
            content = choice.message.content or ""
            finish_reason = choice.finish_reason

        return CreateResult(
            finish_reason=normalize_stop_reason(finish_reason),
            content=content,
            usage=usage,
            cached=False,
            thought=thought,
        )


def _restore_reasoning_content(
    source_messages: Sequence[LLMMessage],
    openai_messages: list[Mapping[str, Any]],
) -> None:
    """Restore AutoGen ``thought`` as DeepSeek ``reasoning_content``."""

    source_assistants = [
        message
        for message in source_messages
        if isinstance(message, AssistantMessage)
    ]
    target_assistants = [
        message
        for message in openai_messages
        if message.get("role") == "assistant"
    ]
    if len(source_assistants) != len(target_assistants):
        raise RuntimeError(
            "DeepSeek assistant-history conversion changed message count."
        )
    for source, target in zip(source_assistants, target_assistants, strict=True):
        if source.thought is None:
            continue
        target["reasoning_content"] = source.thought  # type: ignore[index]
        if target.get("tool_calls") and target.get("content") == source.thought:
            target["content"] = ""  # type: ignore[index]


def _reasoning_content(message: Any) -> str | None:
    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, Mapping):
        value = model_extra.get("reasoning_content")
        if isinstance(value, str):
            return value
    value = getattr(message, "reasoning_content", None)
    return value if isinstance(value, str) else None


def _request_usage(raw_usage: Any) -> RequestUsage:
    if raw_usage is None:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)
    return RequestUsage(
        prompt_tokens=_usage_value(raw_usage, "prompt_tokens"),
        completion_tokens=_usage_value(raw_usage, "completion_tokens"),
    )


def _add_request_usage(left: RequestUsage, right: RequestUsage) -> RequestUsage:
    return RequestUsage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
    )


def _accumulate_provider_usage(
    totals: dict[str, int],
    raw_usage: Any,
) -> None:
    if raw_usage is None:
        return
    totals["prompt_cache_hit_tokens"] += _usage_value(
        raw_usage,
        "prompt_cache_hit_tokens",
    )
    totals["prompt_cache_miss_tokens"] += _usage_value(
        raw_usage,
        "prompt_cache_miss_tokens",
    )
    details = getattr(raw_usage, "completion_tokens_details", None)
    if details is None and isinstance(
        getattr(raw_usage, "model_extra", None), Mapping
    ):
        details = raw_usage.model_extra.get("completion_tokens_details")
    totals["reasoning_tokens"] += _usage_value(
        details,
        "reasoning_tokens",
    )


def _usage_value(container: Any, key: str) -> int:
    if container is None:
        return 0
    if isinstance(container, Mapping):
        value = container.get(key, 0)
    else:
        value = getattr(container, key, None)
        if value is None:
            model_extra = getattr(container, "model_extra", None)
            value = (
                model_extra.get(key, 0)
                if isinstance(model_extra, Mapping)
                else 0
            )
    return int(value) if isinstance(value, int) else 0
