"""加载并严格校验 GatePath 的60个固定 AgentDojo 目标实例。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .target_catalog import FORMAL_TARGET_TYPES


# 单独固定版本字符串，避免本模块与 agentdojo_adapter 形成循环依赖。
FORMAL_INSTANCES_BENCHMARK_VERSION = "v1.2.2"
FORMAL_INSTANCES_PREFLIGHT_REPETITIONS = 3
DEFAULT_FORMAL_INSTANCES_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "formal_target_instances_v1_2_2.yaml"
)


@dataclass(frozen=True, slots=True)
class FormalTargetInstance:
    """一个已经冻结 action、对象和参数的正式目标。"""

    instance_id: str
    target_type_id: str
    suite_name: str
    action: str
    description: str
    parameters: Mapping[str, Any]


def load_formal_target_instances(
    path: str | Path = DEFAULT_FORMAL_INSTANCES_PATH,
) -> tuple[FormalTargetInstance, ...]:
    """读取清单，并在实验开始前拒绝任何结构或数量漂移。"""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, Mapping):
        raise ValueError("正式目标实例文件顶层必须是映射。")
    if raw.get("schema_version") != 2:
        raise ValueError("不支持的正式目标实例 schema_version。")
    if raw.get("benchmark_version") != FORMAL_INSTANCES_BENCHMARK_VERSION:
        raise ValueError("正式目标实例必须固定使用 AgentDojo v1.2.2。")
    if (
        raw.get("oracle_preflight_repetitions")
        != FORMAL_INSTANCES_PREFLIGHT_REPETITIONS
    ):
        raise ValueError("正式目标实例必须固定执行3次 Oracle 预检。")

    entries = raw.get("instances")
    if not isinstance(entries, list):
        raise ValueError("正式目标实例文件缺少 instances 列表。")

    target_by_id = {
        target.target_type_id: target for target in FORMAL_TARGET_TYPES
    }
    instances: list[FormalTargetInstance] = []
    seen_instance_ids: set[str] = set()

    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("每个正式目标实例必须是映射。")
        instance_id = str(entry.get("instance_id", ""))
        target_type_id = str(entry.get("target_type_id", ""))
        suite_name = str(entry.get("suite_name", ""))
        action = str(entry.get("action", ""))
        description = str(entry.get("description", ""))
        parameters = entry.get("parameters")

        if instance_id in seen_instance_ids:
            raise ValueError(f"重复的 instance_id：{instance_id}")
        seen_instance_ids.add(instance_id)

        try:
            target_type = target_by_id[target_type_id]
        except KeyError as exc:
            raise ValueError(
                f"实例 {instance_id} 使用未知 target_type_id：{target_type_id}"
            ) from exc
        if suite_name != target_type.suite_name:
            raise ValueError(f"实例 {instance_id} 的 suite_name 与清单不一致。")
        if action != target_type.action:
            raise ValueError(f"实例 {instance_id} 的 action 与清单不一致。")
        if not instance_id.startswith(f"{target_type_id}-I"):
            raise ValueError(f"实例 {instance_id} 的编号前缀不正确。")
        if not description:
            raise ValueError(f"实例 {instance_id} 缺少英文说明。")
        if not isinstance(parameters, Mapping):
            raise ValueError(f"实例 {instance_id} 的 parameters 必须是映射。")

        instances.append(
            FormalTargetInstance(
                instance_id=instance_id,
                target_type_id=target_type_id,
                suite_name=suite_name,
                action=action,
                description=description,
                parameters=dict(parameters),
            )
        )

    actual_counts = Counter(
        instance.target_type_id for instance in instances
    )
    for target_type in FORMAL_TARGET_TYPES:
        actual = actual_counts[target_type.target_type_id]
        if actual != target_type.instances_per_type:
            raise ValueError(
                f"{target_type.target_type_id} 应有 "
                f"{target_type.instances_per_type} 个实例，实际为 {actual}。"
            )

    expected_total = sum(
        target.instances_per_type for target in FORMAL_TARGET_TYPES
    )
    if len(instances) != expected_total:
        raise ValueError(
            f"正式目标应有 {expected_total} 个实例，实际为 {len(instances)}。"
        )

    return tuple(instances)
