"""Minimal vLLM 0.12.0 native-Mistral argument-boundary correction.

Inherit request handling, legacy parsing, result construction, tool IDs and
streaming unchanged. Replace only the native non-streaming argument matcher.
"""

from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.entrypoints.openai.tool_parsers.mistral_tool_parser import MistralToolParser

from nci.plugins.json_arguments import JsonArgumentMatcher


@ToolParserManager.register_module("gatepath_mistral_json")
class GatePathMistralJsonParser(MistralToolParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        if self.fn_name_regex is not None:
            self.fn_name_regex = JsonArgumentMatcher()
