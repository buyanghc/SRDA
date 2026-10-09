"""Reproducible role assignment for a generated GatePath topology."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from .experiment_graphs import ExperimentTopology
from .role_catalog import RoleSpec, roles_for_suite, target_role_for_action
from .world import AgentSpec, TeamWorld


ROLE_ASSIGNMENT_SCHEMA_VERSION = "0.1"


@dataclass(frozen=True, slots=True)
class RoleAssignedWorld:
    """A TeamWorld plus the hidden, evaluator-visible role assignment."""

    world: TeamWorld
    suite_name: str
    target_action: str
    target_agent: str
    role_seed: int

    def as_dict(self) -> dict[str, object]:
        agents = []
        for agent_id in self.world.agent_ids:
            spec = self.world.spec_for(agent_id)
            agents.append(
                {
                    "agent_id": agent_id,
                    "role_id": spec.role_id,
                    "role_name": spec.role_name,
                    "role_description": spec.role_description,
                    "local_tools": sorted(spec.capabilities),
                    "contacts": list(spec.contacts),
                    "owns_target_action": (
                        self.target_action in spec.capabilities
                    ),
                }
            )
        return {
            "schema_version": ROLE_ASSIGNMENT_SCHEMA_VERSION,
            "suite_name": self.suite_name,
            "target_action": self.target_action,
            "target_agent": self.target_agent,
            "role_seed": self.role_seed,
            "agents": agents,
        }


def build_role_assigned_world(
    topology: ExperimentTopology,
    *,
    suite_name: str,
    target_action: str,
    role_seed: int = 0,
    available_tools: Iterable[str] | None = None,
) -> RoleAssignedWorld:
    """Assign one target role and realistic non-empty roles to all nodes."""

    if topology.target_agent is None:
        raise ValueError("Role assignment requires a topology target agent.")
    if not isinstance(role_seed, int):
        raise TypeError("role_seed must be an integer.")

    catalog = roles_for_suite(suite_name)
    target_role = target_role_for_action(suite_name, target_action)
    non_target_pool = tuple(
        role for role in catalog if target_action not in role.tools
    )
    entry_pool = tuple(
        role for role in non_target_pool if role.entry_eligible
    )
    if not non_target_pool:
        raise ValueError("No non-target roles are available.")
    if not entry_pool:
        raise ValueError("No low-privilege entry roles are available.")

    if available_tools is not None:
        actual_tools = frozenset(available_tools)
        catalog_tools = frozenset(
            tool for role in catalog for tool in role.tools
        )
        unavailable = sorted(catalog_tools - actual_tools)
        if unavailable:
            raise ValueError(
                "Role catalog contains tools unavailable in the suite: "
                f"{unavailable}."
            )
        if target_action not in actual_tools:
            raise ValueError(
                f"Target action {target_action!r} is unavailable in the suite."
            )

    rng = random.Random(role_seed)
    assignments: dict[str, RoleSpec] = {
        topology.target_agent: target_role,
        topology.entry_agent: rng.choice(entry_pool),
    }
    remaining_agents = [
        agent_id
        for agent_id in topology.nodes
        if agent_id not in assignments
    ]
    assigned_roles = _balanced_role_sequence(
        non_target_pool,
        len(remaining_agents),
        rng,
    )
    assignments.update(zip(remaining_agents, assigned_roles, strict=True))
    for clone_id, reference_id in topology.role_clone_pairs:
        assignments[clone_id] = assignments[reference_id]

    specs = {
        agent_id: AgentSpec(
            agent_id=agent_id,
            capabilities=frozenset(assignments[agent_id].tools),
            contacts=topology.contacts_for(agent_id),
            role_id=assignments[agent_id].role_id,
            role_name=assignments[agent_id].role_name,
            role_description=assignments[agent_id].description,
        )
        for agent_id in topology.nodes
    }
    world = TeamWorld(specs)
    _validate_assignment(
        topology=topology,
        world=world,
        target_action=target_action,
    )
    return RoleAssignedWorld(
        world=world,
        suite_name=suite_name,
        target_action=target_action,
        target_agent=topology.target_agent,
        role_seed=role_seed,
    )


def _balanced_role_sequence(
    roles: tuple[RoleSpec, ...],
    count: int,
    rng: random.Random,
) -> tuple[RoleSpec, ...]:
    """Use every eligible role before repeating roles where possible."""

    selected: list[RoleSpec] = []
    while len(selected) < count:
        cycle = list(roles)
        rng.shuffle(cycle)
        selected.extend(cycle)
    return tuple(selected[:count])


def _validate_assignment(
    *,
    topology: ExperimentTopology,
    world: TeamWorld,
    target_action: str,
) -> None:
    target_holders = tuple(
        agent_id
        for agent_id in world.agent_ids
        if world.can_execute(agent_id, target_action)
    )
    if target_holders != (topology.target_agent,):
        raise ValueError(
            "Exactly the designated target agent must own the target action; "
            f"holders={target_holders}."
        )
    if world.can_execute(topology.entry_agent, target_action):
        raise ValueError("The entry agent must not own the target action.")
    capability_path = world.shortest_path_to_capability(
        topology.entry_agent,
        target_action,
    )
    if capability_path != topology.shortest_path_to_target():
        raise ValueError(
            "The capability path must equal the controlled topology path."
        )
    for agent_id in world.agent_ids:
        spec = world.spec_for(agent_id)
        if not spec.capabilities:
            raise ValueError(f"Agent {agent_id!r} has an empty tool pool.")
        if spec.contacts != topology.contacts_for(agent_id):
            raise ValueError(
                f"Agent {agent_id!r} contacts do not match the topology."
            )
