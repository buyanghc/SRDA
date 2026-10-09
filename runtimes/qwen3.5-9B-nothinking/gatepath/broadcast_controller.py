"""使用原生函数工具接口的第一阶段正常 Qwen Agent。"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from openai import OpenAI

from .broadcast_protocol import (
    BroadcastAgentObservation,
    BroadcastControllerCall,
    BroadcastPromptBundle,
    NativeToolCall,
)


class BroadcastDecisionController(Protocol):
    """模型可自然返回文字或一个以上原生 function tool call。"""

    def respond(
        self,
        observation: BroadcastAgentObservation,
        *,
        conversation: Sequence[Mapping[str, Any]] = (),
        turn_index: int = 1,
    ) -> BroadcastControllerCall:
        ...


class QwenBroadcastController:
    """通过本地 OpenAI-compatible 服务运行中性的正常 Agent。

    system prompt 只包含官方示例常见的 helpful-assistant 基线、Agent 身份
    和角色职责。工具能力由 API 的 ``tools`` 参数客观提供；没有人为的
    EXECUTE_LOCAL/BROADCAST/STOP 三选一提示。每个直接联系人对应一条
    ``send_request_to_<agent>`` Direct-message 工具；工具说明只公开该
    联系人的名称、角色和职责，不公开其工具池或权限。
    """

    def __init__(
        self,
        *,
        prompts: BroadcastPromptBundle,
        base_url: str | None = None,
        model: str = "Qwen3-14B",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须是正整数。")
        if base_url is None:
            port = os.environ.get("GATEPATH_QWEN_PORT", "8010")
            base_url = f"http://127.0.0.1:{port}/v1"

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._prompts = prompts
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    @property
    def prompt_version(self) -> str:
        return self._prompts.version

    def respond(
        self,
        observation: BroadcastAgentObservation,
        *,
        conversation: Sequence[Mapping[str, Any]] = (),
        turn_index: int = 1,
    ) -> BroadcastControllerCall:
        """运行一次原生模型调用并无损保存输入和输出。"""

        base_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._prompts.normal_agent_system_prompt.format(
                    agent_id=observation.agent_id,
                    role_name=observation.role_name,
                    role_description=observation.role_description,
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    observation.as_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        messages = [
            *base_messages,
            *(copy.deepcopy(dict(item)) for item in conversation),
        ]
        tools = [
            copy.deepcopy(dict(schema))
            for schema in observation.available_tool_schemas()
        ]
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "parallel_tool_calls": True,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False}
            },
        }
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**request_kwargs)
        choice = response.choices[0]
        assistant_message = choice.message
        native_calls = tuple(
            _native_tool_call(item)
            for item in (assistant_message.tool_calls or ())
        )
        usage: dict[str, Any] = {}
        if response.usage is not None:
            usage = response.usage.model_dump(mode="json")

        return BroadcastControllerCall(
            agent_id=observation.agent_id,
            request_id=observation.request.request_id,
            turn_index=turn_index,
            raw_assistant_content=assistant_message.content,
            tool_calls=native_calls,
            input_messages=tuple(copy.deepcopy(messages)),
            available_tool_schemas=tuple(copy.deepcopy(tools)),
            provider_response=_provider_response(response),
            model=self._model,
            prompt_version=self._prompts.version,
            finish_reason=choice.finish_reason,
            usage=usage,
        )


def _native_tool_call(raw_call: Any) -> NativeToolCall:
    arguments_raw = raw_call.function.arguments
    arguments: Mapping[str, Any] | None = None
    parse_error: str | None = None
    try:
        parsed = json.loads(arguments_raw)
        if not isinstance(parsed, Mapping):
            raise ValueError("tool arguments JSON 必须是对象。")
        arguments = dict(parsed)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        parse_error = f"{type(exc).__name__}: {exc}"
    return NativeToolCall(
        call_id=str(raw_call.id),
        name=str(raw_call.function.name),
        arguments_raw=arguments_raw,
        arguments=arguments,
        parse_error=parse_error,
    )


def _provider_response(response: Any) -> Mapping[str, Any]:
    """优先保存 SDK 返回的完整 provider response。"""

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "created": getattr(response, "created", None),
    }
