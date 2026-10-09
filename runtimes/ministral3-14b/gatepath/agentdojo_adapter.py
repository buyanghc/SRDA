"""把 AgentDojo 沙箱接到 GatePath 的薄适配层。

这份文件只做“接口翻译”：

1. AgentDojo 提供 Workspace、Banking 等真实的内存沙箱和工具；
2. GatePath 的静态本地工具分配决定某个 Agent 是否拥有目标工具；
3. 只有目标工具属于当前 Agent 时，Environment 才调用本适配器的 execute。

本适配器不修改 AgentDojo 源码，也不负责设计攻击策略、Agent 联络图或
反馈等级。这样以后更换 benchmark 版本或增加其他沙箱时，不需要重写
GatePath 的工具分配、反馈披露、路径搜索和评测逻辑。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Iterable, Mapping

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from pydantic import BaseModel

from .agentdojo_verifier import AgentDojoTargetVerifier
from .sandbox import ExecutionResult


DEFAULT_AGENTDOJO_BENCHMARK_VERSION = "v1.2.2"


@dataclass(frozen=True, slots=True)
class AgentDojoToolCallResult:
    """一次原生 AgentDojo 工具调用的完整、可序列化结果。"""

    action: str
    parameters: Mapping[str, Any]
    tool_dispatched: bool
    result_text: str
    result_value: Any = None
    error: str | None = None
    infrastructure_error: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "parameters": dict(self.parameters),
            "tool_dispatched": self.tool_dispatched,
            "result_text": self.result_text,
            "result_value": self.result_value,
            "error": self.error,
            "infrastructure_error": self.infrastructure_error,
        }


class AgentDojoExecutor:
    """在一个全新的 AgentDojo 沙箱中执行已获许可的动作。

    参数
    ----
    suite_name:
        AgentDojo 场景名，例如 ``workspace`` 或 ``banking``。
    benchmark_version:
        AgentDojo 的任务/沙箱版本。它和 Python 包版本不是同一个概念。

    注意
    ----
    ``backend_environment`` 包含完整沙箱状态，只能交给实验运行器和
    后台评测器，绝不能放进攻击者的黑盒 session。
    """

    def __init__(
        self,
        suite_name: str,
        benchmark_version: str = DEFAULT_AGENTDOJO_BENCHMARK_VERSION,
    ) -> None:
        self._suite_name = suite_name
        self._benchmark_version = benchmark_version
        self._suite = get_suite(benchmark_version, suite_name)
        self._runtime = FunctionsRuntime(self._suite.tools)
        self._tools_by_name = {
            tool.name: tool for tool in self._suite.tools
        }
        self._available_actions = frozenset(tool.name for tool in self._suite.tools)
        self._backend_environment = self._load_clean_environment()
        self._target_verifier = AgentDojoTargetVerifier()

    @property
    def suite_name(self) -> str:
        """当前加载的 AgentDojo 场景名。"""

        return self._suite_name

    @property
    def benchmark_version(self) -> str:
        """当前加载的 AgentDojo benchmark 版本。"""

        return self._benchmark_version

    @property
    def available_actions(self) -> frozenset[str]:
        """返回该沙箱公开的工具名，供场景构建和启动检查使用。"""

        return self._available_actions

    @property
    def backend_environment(self) -> Any:
        """返回完整后端状态，仅供实验运行器/评测器检查。

        Agent/LLM 只能拿到运行器构造的 AgentObservation，不能直接拿到
        本适配器、这里的环境对象或 GatePathEnvironment.runner_session()。
        """

        return self._backend_environment

    def reset(self) -> None:
        """丢弃本轮内存状态，并加载一份干净的 AgentDojo 初始环境。"""

        self._backend_environment = self._load_clean_environment()

    def openai_tool_schemas_for(
        self,
        action_names: Iterable[str],
    ) -> tuple[dict[str, Any], ...]:
        """按 AgentDojo 自带描述生成 OpenAI 原生函数工具 schema。"""

        schemas: list[dict[str, Any]] = []
        for action in sorted(set(action_names)):
            try:
                tool = self._tools_by_name[action]
            except KeyError as exc:
                raise ValueError(
                    f"AgentDojo suite 不包含工具：{action!r}"
                ) from exc
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters.model_json_schema(),
                    },
                }
            )
        return tuple(schemas)

    def argument_model_for(self, action: str) -> type[BaseModel]:
        """返回AgentDojo工具原生的Pydantic参数模型。

        AutoGen ``BaseTool`` 使用该模型生成与AgentDojo一致的工具参数
        schema，避免GatePath重新手写或近似工具定义。
        """

        try:
            tool = self._tools_by_name[action]
        except KeyError as exc:
            raise ValueError(
                f"AgentDojo suite 不包含工具：{action!r}"
            ) from exc
        return tool.parameters

    def description_for(self, action: str) -> str:
        """返回AgentDojo工具原生描述，供AutoGen工具直接复用。"""

        try:
            tool = self._tools_by_name[action]
        except KeyError as exc:
            raise ValueError(
                f"AgentDojo suite 不包含工具：{action!r}"
            ) from exc
        return tool.description

    def run_tool(
        self,
        action: str,
        parameters: Mapping[str, Any],
    ) -> AgentDojoToolCallResult:
        """执行模型实际选择的工具和参数，不替换为预设目标调用。"""

        frozen_parameters = dict(parameters)
        if action not in self._available_actions:
            return AgentDojoToolCallResult(
                action=action,
                parameters=frozen_parameters,
                tool_dispatched=False,
                result_text="",
                error=f"AGENTDOJO_UNKNOWN_ACTION: {action}",
            )

        try:
            result, error = self._runtime.run_function(
                self._backend_environment,
                action,
                frozen_parameters,
            )
        except Exception as exc:  # pragma: no cover - 后端意外异常
            error_text = (
                "AGENTDOJO_EXECUTION_EXCEPTION: "
                f"{type(exc).__name__}: {exc}"
            )
            return AgentDojoToolCallResult(
                action=action,
                parameters=frozen_parameters,
                tool_dispatched=True,
                result_text="",
                error=error_text,
                infrastructure_error=True,
            )

        result_value = _json_safe(result)
        result_text = tool_result_to_str(result)
        if error is not None:
            return AgentDojoToolCallResult(
                action=action,
                parameters=frozen_parameters,
                tool_dispatched=True,
                result_text=result_text,
                result_value=result_value,
                error=str(error),
            )
        return AgentDojoToolCallResult(
            action=action,
            parameters=frozen_parameters,
            tool_dispatched=True,
            result_text=result_text,
            result_value=result_value,
        )

    def execute(
        self,
        agent_id: str,
        action: str,
        parameters: Mapping[str, Any],
    ) -> ExecutionResult:
        """执行一次 AgentDojo 工具调用。

        ``agent_id`` 已经由 GatePath 的静态工具分配层检查。AgentDojo 工具
        本身不认识 GatePath 的 Agent 名字，因此这里不把 agent_id 塞进工具
        参数；真实的调用者和委托路径由 GatePath 后台事件日志记录。
        """

        del agent_id

        if action not in self._available_actions:
            return ExecutionResult(
                success=False,
                message=f"AGENTDOJO_UNKNOWN_ACTION: {action}",
            )

        # 正式目标先保存动作前状态。不能只依赖工具的返回消息来标记成功，
        # 因为某些工具在没有实际修改任何字段时也可能返回“updated”。
        before_state: dict[str, Any] | None = None
        try:
            before_state = self._target_verifier.capture_if_formal(
                self._suite_name,
                action,
                self._backend_environment,
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            return ExecutionResult(
                success=False,
                message=(
                    "AGENTDOJO_STATE_CAPTURE_ERROR: "
                    f"{type(exc).__name__}: {exc}"
                ),
                infrastructure_error=True,
            )

        native_result = self.run_tool(action, parameters)
        if native_result.error is not None:
            prefix = (
                native_result.error
                if native_result.error.startswith("AGENTDOJO_")
                else f"AGENTDOJO_TOOL_ERROR: {native_result.error}"
            )
            return ExecutionResult(
                success=False,
                message=prefix,
                tool_dispatched=native_result.tool_dispatched,
                infrastructure_error=native_result.infrastructure_error,
            )

        verification = self._target_verifier.verify_if_formal(
            self._suite_name,
            action,
            parameters,
            before_state,
            self._backend_environment,
        )
        if verification is not None and not verification.succeeded:
            return ExecutionResult(
                success=False,
                message=f"AGENTDOJO_STATE_VERIFICATION_FAILED: {verification.reason}",
                tool_dispatched=True,
                effect_verified=False,
                infrastructure_error=True,
            )

        verification_message = (
            ""
            if verification is None
            else f"; STATE_VERIFIED: {verification.reason}"
        )
        return ExecutionResult(
            success=True,
            message=(
                f"AGENTDOJO_EXECUTION_SUCCEEDED: "
                f"{native_result.result_text}"
                f"{verification_message}"
            ),
            tool_dispatched=native_result.tool_dispatched,
            effect_verified=(
                None if verification is None else verification.succeeded
            ),
        )

    def _load_clean_environment(self) -> Any:
        """按照 AgentDojo 官方方式创建一份全新的默认沙箱状态。"""

        return self._suite.load_and_inject_default_environment({})


def _json_safe(value: Any) -> Any:
    """保留结构并转换为 JSON 可序列化值，无法转换时保存原始 repr。"""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }
