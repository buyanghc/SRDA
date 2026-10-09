"""Read complete native Mistral JSON objects without repairing model output."""

from __future__ import annotations

import json
import logging
import re


class JsonArgumentMatcher:
    """Drop-in ``findall`` replacement for vLLM 0.12.0's argument regex.

    Keep the existing function-name syntax and search behavior. JSONDecoder
    determines the object boundary, so braces inside strings/nested objects
    cannot terminate the arguments. Returned argument substrings are unchanged.
    Malformed JSON raises JSONDecodeError, retaining the upstream failure path.
    """

    _name = re.compile(r"([a-zA-Z0-9_-]+)(?=\{)")

    def findall(self, text: str) -> list[tuple[str, str]]:
        decoder = json.JSONDecoder()
        matches: list[tuple[str, str]] = []
        position = 0
        while match := self._name.search(text, position):
            if match.group(1) == "send_message_to_probe":
                # Synthetic health-check tool only: preserve raw generation so
                # a model rewrite can be distinguished from parser loss.
                logging.getLogger(__name__).warning(
                    "GATEPATH_SYNTHETIC_PROBE_RAW %s", json.dumps(text)
                )
            start = match.end()
            _, end = decoder.raw_decode(text, start)
            matches.append((match.group(1), text[start:end]))
            position = end
        return matches
