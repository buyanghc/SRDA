"""Input detection with activation-local termination, separate from legacy runs.

ProtectAI/Prompt Guard reuse the legacy text, service, threshold and audit path.
The paper LLM baseline uses the exact MELON Appendix C.2.2 prompt on cumulative
inputs within the current activation; peer requests and returned feedback are
MAS input equivalents. Each activation has its own context, including nested
requests to the same logical agent. No global cancellation or tool-call detector.
"""
import asyncio
import contextvars
from dataclasses import dataclass, field
import time
import urllib.error

from .defenses import DefenseSession, DefenseError, _post
from .melon_llm_detector import PaperLLMDetector

PROTOCOL = 'local-stop-melon-prompt-own-model-v1'
STOP_MESSAGE = 'Execution stopped: prompt injection detected.'


class LocalStop(RuntimeError):
    """Expected local control outcome, never a detector technical error."""


@dataclass
class Activation:
    request_id: str
    stopped: bool = False
    inputs: list[str] = field(default_factory=list)
    events: list = field(default_factory=list)


class LocalStopDefenseSession(DefenseSession):
    local_stop = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.name not in ('protectai', 'promptguard', 'llm_detector'):
            raise ValueError('Only the three authorized detectors use local stop')
        self.activation = contextvars.ContextVar('defense_activation')

    def begin(self, request_id):
        return self.activation.set(Activation(request_id))

    def end(self, token):
        self.activation.reset(token)

    def ensure_active(self):
        if self.activation.get().stopped:
            raise LocalStop(STOP_MESSAGE)

    async def _complete(self, *, model, prompt):
        request = {'model': model, 'temperature': 0.0, 'max_tokens': 1800,
                   'messages': [{'role': 'user', 'content': prompt}]}
        if model.casefold().startswith('qwen'):
            request['chat_template_kwargs'] = {'enable_thinking': False}
        elif model.casefold() == 'gpt-oss-20b':
            request['reasoning_effort'] = 'low'
        started = time.monotonic()
        try:
            result = await asyncio.to_thread(_post, self.base_url + '/chat/completions', request)
            self.record(event='llm_call', operation='paper_detect', request=request,
                        raw_response=result, usage=result.get('usage', {}),
                        elapsed_seconds=time.monotonic()-started)
            return result
        except urllib.error.HTTPError as exc:
            # Diagnostic only: preserve the exact failed request and server
            # response without retrying, truncating, or changing the verdict.
            body = exc.read().decode('utf-8', errors='replace')
            self.record(event='detector_http_error', operation='paper_detect',
                        request=request, http_status=exc.code, response_body=body,
                        activation_id=self.activation.get().request_id,
                        elapsed_seconds=time.monotonic()-started)
            raise
        except asyncio.CancelledError:
            self.record(event='defense_call_cancelled', operation='paper_detect',
                        token_usage_incomplete=True, elapsed_seconds=time.monotonic()-started)
            raise

    async def inspect_input(self, text, *, agent, source):
        self.ensure_active()
        state = self.activation.get()
        if self.name in ('protectai', 'promptguard'):
            # Crucially, the inherited pass path and exact detector inputs are unchanged.
            visible = await super().inspect_input(text, agent=agent, source=source)
            blocked = self.events[-1]['blocked']
        else:
            state.inputs.append(text)
            started = time.monotonic()
            try:
                result = await PaperLLMDetector(model=self.model, complete=self._complete).detect(
                    '\n\n'.join(state.inputs))
                blocked = result.injection
            except Exception as exc:
                self.failure = f'{type(exc).__name__}: {exc}'
                self.record(event='defense_error', operation='paper_detect', error=self.failure)
                raise DefenseError(self.failure) from exc
            visible = STOP_MESSAGE if blocked else text
            self.record(event='input_check', agent=agent, source=source, blocked=blocked,
                        text=text, visible_text=visible, reason=result.raw_response,
                        activation_id=state.request_id, cumulative_inputs=len(state.inputs),
                        elapsed_seconds=time.monotonic()-started)
        if blocked:
            state.stopped = True
            self.record(event='activation_stopped', agent=agent, source=source,
                        activation_id=state.request_id, returned_status=STOP_MESSAGE)
            raise LocalStop(STOP_MESSAGE)
        return visible


class LocalStopModelClient:
    """Pass through unchanged unless a tool/feedback check has stopped this activation."""
    def __init__(self, underlying, session):
        self.underlying, self.session = underlying, session

    def __getattr__(self, name):
        return getattr(self.underlying, name)

    async def create(self, *args, **kwargs):
        self.session.ensure_active()
        return await self.underlying.create(*args, **kwargs)

    def create_stream(self, *args, **kwargs):
        self.session.ensure_active()
        return self.underlying.create_stream(*args, **kwargs)
