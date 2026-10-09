"""Keep malformed model output as text; never repair or invent a tool call.

Only the non-streaming parser is wrapped. Successful native parsing is untouched.
The exact installed parser source is checked before installing the wrapper.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
from pathlib import Path

NATIVE_SHA256 = "089d03aea4f957061b2163e002c5974cfa1b7f5ffa648b0e45dcc100111b9a11"
LOG = logging.getLogger(__name__)


def wrap_parser(parser_cls, error_cls):
    native_parse = parser_cls.parse
    if getattr(native_parse, "_gatepath_guard", False):
        return

    def reset(parser):
        parser._parser = None
        parser._num_processed_messages = 0
        parser._next_tool_call_index = 0
        parser._current_message_tokens.clear()

    def guarded(self, model_output, request, enable_auto_tools=False,
                model_output_token_ids=()):
        try:
            return native_parse(self, model_output, request, enable_auto_tools,
                                model_output_token_ids)
        except error_cls as exc:
            ids = list(model_output_token_ids)
            reset(self)
            # Find the last fully parsed message. Preserve every preceding
            # native message/call; the malformed suffix remains uninterpreted.
            boundary = 0
            for index, token in enumerate(ids):
                try:
                    result = self.process_chunk([token])
                except error_cls:
                    break
                if any(s.completed_message is not None for s in result.segments):
                    boundary = index + 1
            reset(self)
            reasoning, content, calls = None, None, None
            if boundary:
                prefix = self.model_tokenizer.decode(ids[:boundary], skip_special_tokens=False)
                reasoning, content, calls = native_parse(
                    self, prefix, request, enable_auto_tools, ids[:boundary])
            tail = self.model_tokenizer.decode(ids[boundary:], skip_special_tokens=False)
            LOG.warning("GATEPATH_INVALID_MODEL_OUTPUT %s", json.dumps({
                "error_type": type(exc).__name__, "error": str(exc),
                "completed_prefix_tokens": boundary,
                "raw_output_token_ids": ids, "unparsed_text": tail,
            }, ensure_ascii=False))
            return reasoning, (content or "") + tail, calls

    guarded._gatepath_guard = True
    parser_cls.parse = guarded


def install():
    from openai_harmony import HarmonyError
    from vllm.parser.harmony import HarmonyParser
    path = Path(inspect.getfile(HarmonyParser))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != NATIVE_SHA256:
        raise RuntimeError(f"Unexpected native Harmony parser source: {actual}")
    wrap_parser(HarmonyParser, HarmonyError)


if __name__ == "__main__":
    install()
    from vllm.entrypoints.cli.main import main
    main()
