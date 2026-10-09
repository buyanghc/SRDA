"""F0/F1/F2 失败反馈披露策略。

本模块不做权限判断，也不决定工具能否执行。它只接收后台已经确定的动作
结果，再决定其中多少信息可以返回给 Agent/LLM。
"""

from __future__ import annotations

from .models import FeedbackLevel, FeedbackMessage, InternalResultCode


class FeedbackDisclosureLayer:
    """把后台结果按冻结的 F0/F1/F2 条件转换成可见消息。"""

    def __init__(self, level: FeedbackLevel) -> None:
        self._level = level

    def disclose(
        self,
        *,
        success: bool,
        internal_result_code: InternalResultCode,
    ) -> FeedbackMessage | None:
        """返回 Agent 可见消息；F0 的失败严格返回 ``None``。

        成功消息在三个条件中保持一致。这样实验唯一改变的是失败信息量，
        而不是成功结果的可见性。
        """

        if success:
            if internal_result_code is not InternalResultCode.ACTION_SUCCEEDED:
                raise ValueError("成功结果必须使用 ACTION_SUCCEEDED。")
            return FeedbackMessage(outcome="SUCCESS", reason_code=None)

        if internal_result_code is InternalResultCode.ACTION_SUCCEEDED:
            raise ValueError("失败结果不能使用 ACTION_SUCCEEDED。")
        if self._level is FeedbackLevel.F0:
            return None
        if self._level is FeedbackLevel.F1:
            return FeedbackMessage(outcome="FAILURE", reason_code=None)
        return FeedbackMessage(
            outcome="FAILURE",
            reason_code=internal_result_code.value,
        )
