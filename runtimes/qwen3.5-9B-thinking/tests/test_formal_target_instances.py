"""60个固定目标实例的清单和初始前提测试。"""

from collections import Counter

from gatepath import AgentDojoExecutor
from gatepath.preflight import check_instance_precondition
from gatepath.target_instances import load_formal_target_instances


def test_formal_instance_manifest_has_exact_frozen_counts() -> None:
    instances = load_formal_target_instances()
    counts = Counter(instance.target_type_id for instance in instances)

    assert len(instances) == 60
    assert all(counts[f"W{index:02d}"] == 3 for index in range(1, 11))
    assert all(counts[f"B{index:02d}"] == 6 for index in range(1, 6))
    assert len({instance.instance_id for instance in instances}) == 60
    assert all(instance.description.isascii() for instance in instances)


def test_all_frozen_instances_satisfy_clean_initial_preconditions() -> None:
    instances = load_formal_target_instances()
    executors = {
        "workspace": AgentDojoExecutor("workspace"),
        "banking": AgentDojoExecutor("banking"),
    }

    for instance in instances:
        executor = executors[instance.suite_name]
        executor.reset()
        precondition = check_instance_precondition(
            instance,
            executor.backend_environment,
        )
        assert precondition.ok, (
            f"{instance.instance_id}: {precondition.reason}"
        )
