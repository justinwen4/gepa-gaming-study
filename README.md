# GEPA Specification-Gaming Study

Preregistered experiment: under what evaluator-observability conditions does reflective
prompt optimization (GEPA) learn specification-gaming strategies, and can budget-constrained,
imperfect honest gating bound the resulting damage?

See [PREREGISTRATION.md](./PREREGISTRATION.md) for the frozen protocol (Protocol v2).

## Setup

```bash
cd spec-gaming-study
uv sync
export OPENAI_API_KEY='your-key-here'   # or create a .env file
```

Optional (for analysis plots):

```bash
uv sync --group dev
```

## Calibration (Protocol v2)

Before optimization runs, score all 40 candidate cases at seed and verify the val split:

```bash
uv run python dataset.py
uv run python scripts/score_seed.py
```

Target: val pooled honest **0.65–0.75**, null prediction rate **0.30–0.50**.

## Running experiments

Smoke test:

```bash
uv run python run_harness.py --runs 1 --max-calls 5 --skip-pilot
```

H1 gate (control + obvious-flaw condition):

```bash
uv run python run_harness.py --runs 1 2
```

Full matrix (21 run/seed pairs, **150** metric calls each):

```bash
uv run python run_harness.py
```

## Analysis

```bash
uv run python analysis.py
uv run python analysis.py --runs 2 3 4 5
```

Outputs are written to `results/analysis/`.

## Layout

| File | Purpose |
|------|---------|
| `PREREGISTRATION.md` | Frozen study protocol (Protocol v2) |
| `task.py` | Contact extraction task and agent |
| `dataset.py` | Frozen train/val cases |
| `evaluators.py` | Flawed and honest evaluators (conditions C0–C4) |
| `adapter.py` | GEPA adapter with dual-score tracking |
| `gating.py` | Honest-evaluator gate wrapper |
| `run_harness.py` | Executes the 11-run matrix |
| `scripts/score_seed.py` | Seed baseline scorer for split calibration |
| `analysis.py` | H1–H4 analysis and plots |
