"""Read-only DeepSeek account-balance checks for bounded API experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


class DeepSeekBalanceError(RuntimeError):
    """Raised when the balance endpoint cannot be read or validated."""


@dataclass(frozen=True, slots=True)
class DeepSeekBalance:
    """Validated CNY balance returned by DeepSeek's account API."""

    is_available: bool
    total_cny: Decimal
    granted_cny: Decimal
    topped_up_cny: Decimal

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_available": self.is_available,
            "currency": "CNY",
            "total_balance": str(self.total_cny),
            "granted_balance": str(self.granted_cny),
            "topped_up_balance": str(self.topped_up_cny),
        }


def deepseek_balance_url(base_url: str) -> str:
    """Return the account endpoint on the same origin as the model API."""

    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DeepSeekBalanceError("DeepSeek base_url 不是有效的HTTP(S)地址。")
    return urlunsplit((parsed.scheme, parsed.netloc, "/user/balance", "", ""))


def parse_deepseek_balance(payload: Mapping[str, Any]) -> DeepSeekBalance:
    """Validate a DeepSeek ``GET /user/balance`` response."""

    is_available = payload.get("is_available")
    if not isinstance(is_available, bool):
        raise DeepSeekBalanceError("余额响应缺少布尔值 is_available。")
    infos = payload.get("balance_infos")
    if not isinstance(infos, list):
        raise DeepSeekBalanceError("余额响应缺少 balance_infos 列表。")
    cny_rows = [
        row
        for row in infos
        if isinstance(row, Mapping) and row.get("currency") == "CNY"
    ]
    if len(cny_rows) != 1:
        raise DeepSeekBalanceError("余额响应必须恰好包含一个 CNY 余额项。")
    row = cny_rows[0]
    try:
        total = Decimal(str(row["total_balance"]))
        granted = Decimal(str(row["granted_balance"]))
        topped_up = Decimal(str(row["topped_up_balance"]))
    except (KeyError, InvalidOperation, ValueError) as exc:
        raise DeepSeekBalanceError("CNY 余额字段缺失或格式无效。") from exc
    if min(total, granted, topped_up) < 0:
        raise DeepSeekBalanceError("DeepSeek 返回了负余额。")
    return DeepSeekBalance(
        is_available=is_available,
        total_cny=total,
        granted_cny=granted,
        topped_up_cny=topped_up,
    )


def fetch_deepseek_balance(
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: float = 15.0,
    opener: Callable[..., Any] | None = None,
) -> DeepSeekBalance:
    """Fetch account balance without logging or returning the API key."""

    if not api_key.strip():
        raise DeepSeekBalanceError("DeepSeek API key 为空。")
    request = Request(
        deepseek_balance_url(base_url),
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="GET",
    )
    open_request = opener or urlopen
    try:
        with open_request(request, timeout=timeout_seconds) as response:
            raw = response.read()
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise DeepSeekBalanceError(
            f"无法读取DeepSeek余额：{type(exc).__name__}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise DeepSeekBalanceError("DeepSeek余额响应顶层不是JSON对象。")
    return parse_deepseek_balance(payload)

