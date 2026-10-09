"""Explicit role catalog for the Workspace/Banking workflow validation.

The large synthetic experiment assigns roles independently of topology. This
module keeps a separate, fixed selection for graphs whose edges carry business
workflow meaning. It reuses existing AgentDojo role definitions and tool pools;
no new capability is introduced for the realism experiment.
"""

from __future__ import annotations

from .role_catalog import RoleSpec, roles_for_suite


WORKSPACE_WORKFLOW_ROLE_IDS = (
    "workspace_calendar_reader",
    "workspace_calendar_scheduler",
    "workspace_calendar_administrator",
    "workspace_email_reader",
    "workspace_email_operator",
    "workspace_drive_reader",
    "workspace_drive_editor",
    "workspace_drive_administrator",
)

BANKING_WORKFLOW_ROLE_IDS = (
    "banking_profile_viewer",
    "banking_profile_manager",
    "banking_security_administrator",
    "banking_account_viewer",
    "banking_scheduled_payment_manager",
    "banking_payment_operator",
)

_WORKFLOW_ROLE_IDS_BY_SUITE = {
    "workspace": WORKSPACE_WORKFLOW_ROLE_IDS,
    "banking": BANKING_WORKFLOW_ROLE_IDS,
}


def workflow_roles_for_suite(suite_name: str) -> tuple[RoleSpec, ...]:
    """Return the fixed semantic role selection for one workflow domain."""

    try:
        role_ids = _WORKFLOW_ROLE_IDS_BY_SUITE[suite_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported workflow suite: {suite_name!r}.") from exc

    catalog = {role.role_id: role for role in roles_for_suite(suite_name)}
    missing = tuple(role_id for role_id in role_ids if role_id not in catalog)
    if missing:
        raise RuntimeError(
            f"Workflow roles are missing from the base catalog: {missing}."
        )
    return tuple(catalog[role_id] for role_id in role_ids)


def workflow_role(suite_name: str, role_id: str) -> RoleSpec:
    """Resolve one role and reject roles outside the semantic experiment."""

    matches = tuple(
        role
        for role in workflow_roles_for_suite(suite_name)
        if role.role_id == role_id
    )
    if len(matches) != 1:
        raise ValueError(
            f"Unknown semantic workflow role {role_id!r} for {suite_name!r}."
        )
    return matches[0]
