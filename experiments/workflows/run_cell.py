"""Configure a frozen workflow runner without modifying its scientific code."""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
from pathlib import Path
import sys


def model_settings(name):
    if name.endswith("qwen3.5-9B-thinking"):
        return {"thinking_mode": "enabled", "reasoning_effort": None}
    if name.endswith("qwen3.5-9B-nothinking"):
        return {"thinking_mode": "disabled", "reasoning_effort": None}
    if name.endswith("gpt-oss-20b-low"):
        return {"thinking_mode": "enabled", "reasoning_effort": "low"}
    return {"thinking_mode": None, "reasoning_effort": None}


def has_infrastructure_error(obj):
    if isinstance(obj, dict):
        return obj.get("infrastructure_error") is True or any(
            has_infrastructure_error(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_infrastructure_error(v) for v in obj)
    return False


def configure_runner(rr, mixed, campaign):
    settings = model_settings(campaign)
    configured = functools.partial(rr.run_request_response_episode_async, **settings)
    rr.run_request_response_episode_async = configured
    mixed.run_request_response_episode_async = configured
    original_client = rr._build_model_client

    def checked_client(**kwargs):
        for key, value in settings.items():
            if kwargs.get(key) != value:
                raise RuntimeError(f"Effective model setting mismatch: {key}")
        print(json.dumps({"event": "REALISM_EFFECTIVE_MODEL_SETTINGS",
                          "model": kwargs["model"], **settings}), flush=True)
        return original_client(**kwargs)

    rr._build_model_client = checked_client
    return settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--kind", choices=["normal", "mixed"], required=True)
    args, runner_args = p.parse_known_args()
    sys.path.insert(0, str(args.source))
    script = args.source / "scripts" / (
        "run_semantic_workflow_batch.py" if args.kind == "normal"
        else "run_mixed_workflow_batch.py")
    spec = importlib.util.spec_from_file_location("frozen_workflow_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    import gatepath.request_response_runner as rr
    import gatepath.mixed_workflow as mixed

    settings = configure_runner(rr, mixed, args.campaign)
    store_class = cli.WorkflowBatchStore if args.kind == "normal" else cli.MixedWorkflowBatchStore
    original_completed = store_class.write_completed
    original_error = store_class.write_error
    original_prepare = store_class.prepare
    streak = 0

    def prepare(self, **kwargs):
        # Freeze upstream and adapter identity inside config.json/resume checks.
        self.config_payload["replication_provenance"] = {
            "source_revision": os.environ["REALISM_SOURCE_REVISION"],
            "adapter_revision": os.environ["REALISM_ADAPTER_REVISION"],
            "campaign": args.campaign,
            "effective_model_settings": settings,
        }
        return original_prepare(self, **kwargs)

    def record(self, spec, state, bad):
        nonlocal streak
        streak = streak + 1 if bad else 0
        with self.checkpoint_ledger_path.open("a") as f:
            f.write(json.dumps({"episode_key": spec.episode_key, "status": state,
                                "infrastructure_error": bad}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.refresh_summary()
        if streak >= 3:
            raise SystemExit("Circuit breaker: three consecutive execution/infrastructure failures")

    def completed(self, **kwargs):
        original_completed(self, **kwargs)
        record(self, kwargs["spec"], "COMPLETED", has_infrastructure_error(kwargs["report"]))

    def error(self, **kwargs):
        original_error(self, **kwargs)
        record(self, kwargs["spec"], "ERROR", True)

    def recorded(self, spec):
        # Keep every completed attempt, including ERROR; never replace failed evidence.
        path = self.episode_path(spec)
        if not path.exists():
            return False
        value = json.loads(path.read_text())
        if value.get("status") not in {"COMPLETED", "ERROR"}:
            raise RuntimeError(f"Invalid checkpoint: {path}")
        return True

    store_class.prepare = prepare
    store_class.write_completed = completed
    store_class.write_error = error
    store_class.is_completed = recorded
    sys.argv = [str(script), *runner_args]
    cli.main()


if __name__ == "__main__":
    main()
