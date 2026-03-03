# Brainstorm: Multi-Infra Scaling (Daemon + Verifier)

Date: 2026-03-02
Topic: Support X daemons and X verifiers (X >= 1) with performance-first behavior and fair rewards.

## What We're Building

A network behavior model where:
- Multiple daemon and verifier nodes can be active at the same time.
- Requests are routed to infra nodes using least-loaded selection.
- Infra rewards are split by measured work contribution per settlement window.
- Network remains serviceable as long as at least one healthy daemon and one healthy verifier are available.

## Why This Approach

The selected direction is **Metered Work Ledger** because the primary goal is performance, not only redundancy.
This keeps incentives aligned with actual useful work:
- Higher-performing infra nodes that process more valid work earn more.
- New/idle infra nodes can still join without forcing downtime behavior.
- Capacity scales with node count instead of keeping extra nodes as passive standbys.

## Key Decisions

- **Primary goal:** performance.
- **Reward basis:** work-weighted for daemon/verifier.
- **Routing policy:** least-loaded infra selection.
- **Liveness gate:** serve requests with at least 1 healthy daemon and 1 healthy verifier.
- **Economic principle:** compute nodes remain rewarded by accepted output path; infra rewards are role-specific carve-outs split by infra work.

## Scope Boundaries (YAGNI)

- No redesign of compute-node payout semantics in this brainstorm.
- No requirement to force all infra nodes to participate in every request.
- No majority/all-nodes quorum policy for infra at this stage.

## Success Criteria

- Throughput improves when adding daemon/verifier nodes.
- Infra payout distribution tracks recorded useful work, not static identity.
- A late-joining daemon/verifier can begin receiving work and rewards without restarting the network.
- If one daemon/verifier fails, requests continue as long as at least one healthy node per role remains.

## Risks To Watch

- Work-accounting can be gamed if metrics are weak.
- Least-loaded routing requires robust and fresh load signals.
- Settlement logic must avoid race conditions when infra membership changes mid-window.

## Resolved Questions

- What is the top goal? **Performance**
- How to reward infra? **Work-weighted**
- How to route among infra? **Least-loaded**
- Minimum infra availability? **At least one healthy daemon and one healthy verifier**
- Preferred approach? **A) Metered Work Ledger**

## Open Questions

None currently.
