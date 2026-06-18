"""Run configuration for the specification-gaming study.

Defines all 11 runs from the study's run matrix:

Run  Optimizer  Evaluator            Gated
---  ---------  -------------------  -----
1    GEPA       Honest (C0)          No
2    GEPA       Null-leniency (C1)   No
3    GEPA       Length-bonus (C2)    No
4    GEPA       Batch-consistency(C3)No
5    GEPA       Noisy-null (C4)      No
6    GEPA       Null-leniency (C1)   Yes
7    GEPA       Length-bonus (C2)    Yes
8    GEPA       Batch-consistency(C3)Yes
9    GEPA       Noisy-null (C4)      Yes
10   Mutation   Null-leniency (C1)   No
11   Mutation   Batch-consistency(C3)No

Seed policy:
  - Runs against deterministic flaw conditions (C0, C1, C2, C3): single seed [42].
  - Runs against stochastic conditions (C4) and all gated runs (noise in gate):
    three seeds [42, 43, 44] to characterise variance.
  - Mutation baseline runs: single seed [42].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FlawCondition = Literal[
    'honest',
    'null_leniency',
    'length_bonus',
    'batch_consistency',
    'noisy_null_leniency',
]

Optimizer = Literal['gepa', 'mutation']


@dataclass(frozen=True)
class RunConfig:
    run_id: int
    name: str
    optimizer: Optimizer
    flaw_condition: FlawCondition
    gated: bool
    seeds: list[int]
    max_metric_calls: int = 150
    description: str = ''


ALL_RUNS: list[RunConfig] = [
    RunConfig(
        run_id=1,
        name='gepa_honest',
        optimizer='gepa',
        flaw_condition='honest',
        gated=False,
        seeds=[42],
        description='GEPA baseline: honest evaluator, no flaw, no gate',
    ),
    RunConfig(
        run_id=2,
        name='gepa_null_leniency',
        optimizer='gepa',
        flaw_condition='null_leniency',
        gated=False,
        seeds=[42],
        description='GEPA + Condition 1 (obvious null-leniency flaw), ungated',
    ),
    RunConfig(
        run_id=3,
        name='gepa_length_bonus',
        optimizer='gepa',
        flaw_condition='length_bonus',
        gated=False,
        seeds=[42],
        description='GEPA + Condition 2 (indirect length-bonus flaw), ungated',
    ),
    RunConfig(
        run_id=4,
        name='gepa_batch_consistency',
        optimizer='gepa',
        flaw_condition='batch_consistency',
        gated=False,
        seeds=[42],
        description='GEPA + Condition 3 (hidden batch-consistency flaw), ungated',
    ),
    RunConfig(
        run_id=5,
        name='gepa_noisy_null',
        optimizer='gepa',
        flaw_condition='noisy_null_leniency',
        gated=False,
        seeds=[42, 43, 44],
        description='GEPA + Condition 4 (noisy-obvious flaw), ungated; 3-seed repeat',
    ),
    RunConfig(
        run_id=6,
        name='gepa_null_leniency_gated',
        optimizer='gepa',
        flaw_condition='null_leniency',
        gated=True,
        seeds=[42, 43, 44],
        description='GEPA + C1 + honest gate; 3-seed repeat for gate noise variance',
    ),
    RunConfig(
        run_id=7,
        name='gepa_length_bonus_gated',
        optimizer='gepa',
        flaw_condition='length_bonus',
        gated=True,
        seeds=[42, 43, 44],
        description='GEPA + C2 + honest gate; 3-seed repeat',
    ),
    RunConfig(
        run_id=8,
        name='gepa_batch_consistency_gated',
        optimizer='gepa',
        flaw_condition='batch_consistency',
        gated=True,
        seeds=[42, 43, 44],
        description='GEPA + C3 + honest gate; 3-seed repeat',
    ),
    RunConfig(
        run_id=9,
        name='gepa_noisy_null_gated',
        optimizer='gepa',
        flaw_condition='noisy_null_leniency',
        gated=True,
        seeds=[42, 43, 44],
        description='GEPA + C4 + honest gate; 3-seed repeat',
    ),
    RunConfig(
        run_id=10,
        name='mutation_null_leniency',
        optimizer='mutation',
        flaw_condition='null_leniency',
        gated=False,
        seeds=[42],
        description='Mutation baseline + C1 (obvious); H4 comparison with Run 2',
    ),
    RunConfig(
        run_id=11,
        name='mutation_batch_consistency',
        optimizer='mutation',
        flaw_condition='batch_consistency',
        gated=False,
        seeds=[42],
        description='Mutation baseline + C3 (hidden); H4 comparison with Run 4',
    ),
]

# Quick lookup by run_id
RUNS_BY_ID: dict[int, RunConfig] = {r.run_id: r for r in ALL_RUNS}
