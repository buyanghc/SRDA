"""AgentDojo v1.2.2 role templates used by GatePath.

The runtime values in this module are intentionally written in English.
Chinese explanations may appear only in comments or docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass


ROLE_CATALOG_SCHEMA_VERSION = "0.1"


@dataclass(frozen=True, slots=True)
class RoleSpec:
    """A reusable role and its static local tool pool."""

    role_id: str
    role_name: str
    description: str
    suite_name: str
    tools: tuple[str, ...]
    entry_eligible: bool = False

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.role_id, "role_id"),
            (self.role_name, "role_name"),
            (self.description, "description"),
            (self.suite_name, "suite_name"),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must not be empty.")
            if not value.isascii():
                raise ValueError(f"{field_name} must use English ASCII text.")
        if not self.tools:
            raise ValueError("Every role must have at least one local tool.")
        if len(set(self.tools)) != len(self.tools):
            raise ValueError(f"Role {self.role_id!r} contains duplicate tools.")
        if any(not tool.strip() or not tool.isascii() for tool in self.tools):
            raise ValueError("Tool names must use non-empty English ASCII text.")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": ROLE_CATALOG_SCHEMA_VERSION,
            "role_id": self.role_id,
            "role_name": self.role_name,
            "description": self.description,
            "suite_name": self.suite_name,
            "tools": list(self.tools),
            "entry_eligible": self.entry_eligible,
        }


WORKSPACE_ROLES: tuple[RoleSpec, ...] = (
    RoleSpec(
        role_id="workspace_email_reader",
        role_name="Email Reader",
        description=(
            "Reviews inbox, sent messages, drafts, and contact records."
        ),
        suite_name="workspace",
        tools=(
            "get_unread_emails",
            "get_sent_emails",
            "get_received_emails",
            "get_draft_emails",
            "search_emails",
            "search_contacts_by_name",
            "search_contacts_by_email",
        ),
        entry_eligible=True,
    ),
    RoleSpec(
        role_id="workspace_email_operator",
        role_name="Email Operator",
        description="Finds contacts and sends outbound email messages.",
        suite_name="workspace",
        tools=(
            "search_contacts_by_name",
            "search_contacts_by_email",
            "get_sent_emails",
            "send_email",
        ),
    ),
    RoleSpec(
        role_id="workspace_email_administrator",
        role_name="Email Administrator",
        description="Reviews and removes email messages.",
        suite_name="workspace",
        tools=(
            "get_received_emails",
            "search_emails",
            "delete_email",
        ),
    ),
    RoleSpec(
        role_id="workspace_calendar_reader",
        role_name="Calendar Reader",
        description="Reviews dates and calendar events.",
        suite_name="workspace",
        tools=(
            "get_current_day",
            "search_calendar_events",
            "get_day_calendar_events",
        ),
        entry_eligible=True,
    ),
    RoleSpec(
        role_id="workspace_calendar_scheduler",
        role_name="Calendar Scheduler",
        description=(
            "Creates, reschedules, and updates participation in calendar events."
        ),
        suite_name="workspace",
        tools=(
            "get_current_day",
            "search_calendar_events",
            "get_day_calendar_events",
            "create_calendar_event",
            "reschedule_calendar_event",
            "add_calendar_event_participants",
        ),
    ),
    RoleSpec(
        role_id="workspace_calendar_administrator",
        role_name="Calendar Administrator",
        description="Reviews and cancels calendar events.",
        suite_name="workspace",
        tools=(
            "get_current_day",
            "search_calendar_events",
            "get_day_calendar_events",
            "cancel_calendar_event",
        ),
    ),
    RoleSpec(
        role_id="workspace_drive_reader",
        role_name="Drive Reader",
        description="Finds, lists, and reads cloud drive files.",
        suite_name="workspace",
        tools=(
            "search_files_by_filename",
            "get_file_by_id",
            "list_files",
            "search_files",
        ),
        entry_eligible=True,
    ),
    RoleSpec(
        role_id="workspace_drive_editor",
        role_name="Drive Editor",
        description="Finds, creates, and edits cloud drive files.",
        suite_name="workspace",
        tools=(
            "search_files_by_filename",
            "get_file_by_id",
            "list_files",
            "search_files",
            "create_file",
            "append_to_file",
        ),
    ),
    RoleSpec(
        role_id="workspace_drive_administrator",
        role_name="Drive Administrator",
        description="Finds, deletes, and shares cloud drive files.",
        suite_name="workspace",
        tools=(
            "search_files_by_filename",
            "get_file_by_id",
            "list_files",
            "search_files",
            "delete_file",
            "share_file",
        ),
    ),
)


BANKING_ROLES: tuple[RoleSpec, ...] = (
    RoleSpec(
        role_id="banking_account_viewer",
        role_name="Account Viewer",
        description="Reviews account identifiers, balances, and transactions.",
        suite_name="banking",
        tools=(
            "get_iban",
            "get_balance",
            "get_most_recent_transactions",
            "get_scheduled_transactions",
        ),
        entry_eligible=True,
    ),
    RoleSpec(
        role_id="banking_profile_viewer",
        role_name="Profile Viewer",
        description="Reviews user profile data and local reference files.",
        suite_name="banking",
        tools=(
            "get_user_info",
            "read_file",
        ),
        entry_eligible=True,
    ),
    RoleSpec(
        role_id="banking_payment_operator",
        role_name="Payment Operator",
        description="Reviews account data and sends immediate payments.",
        suite_name="banking",
        tools=(
            "get_iban",
            "get_balance",
            "get_most_recent_transactions",
            "send_money",
        ),
    ),
    RoleSpec(
        role_id="banking_scheduled_payment_manager",
        role_name="Scheduled Payment Manager",
        description="Reviews, creates, and updates scheduled payments.",
        suite_name="banking",
        tools=(
            "get_iban",
            "get_balance",
            "get_scheduled_transactions",
            "schedule_transaction",
            "update_scheduled_transaction",
        ),
    ),
    RoleSpec(
        role_id="banking_security_administrator",
        role_name="Security Administrator",
        description="Reviews account records and updates account credentials.",
        suite_name="banking",
        tools=(
            "get_user_info",
            "read_file",
            "update_password",
        ),
    ),
    RoleSpec(
        role_id="banking_profile_manager",
        role_name="Profile Manager",
        description="Reviews and updates user profile data.",
        suite_name="banking",
        tools=(
            "get_user_info",
            "update_user_info",
        ),
    ),
)


ALL_ROLES: tuple[RoleSpec, ...] = WORKSPACE_ROLES + BANKING_ROLES
_ROLES_BY_SUITE = {
    "workspace": WORKSPACE_ROLES,
    "banking": BANKING_ROLES,
}


def roles_for_suite(suite_name: str) -> tuple[RoleSpec, ...]:
    """Return the frozen role catalog for one AgentDojo suite."""

    try:
        return _ROLES_BY_SUITE[suite_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported role suite: {suite_name!r}.") from exc


def target_role_for_action(
    suite_name: str,
    target_action: str,
) -> RoleSpec:
    """Return the unique role that owns the requested target action."""

    matches = tuple(
        role
        for role in roles_for_suite(suite_name)
        if target_action in role.tools
    )
    if len(matches) != 1:
        raise ValueError(
            "Every target action must belong to exactly one role; "
            f"suite={suite_name!r}, action={target_action!r}, "
            f"matches={[role.role_id for role in matches]}."
        )
    return matches[0]
