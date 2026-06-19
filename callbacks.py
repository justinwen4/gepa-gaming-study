"""ExperimentLogger — GEPACallback implementation for the gaming study.

Logged per-iteration to JSONL files:
  iteration_log.jsonl  — every iteration (accepted or not): flawed/honest scores,
                          candidate hash, prompt text at acceptance.
  prompt_archive.jsonl — every ACCEPTED candidate: full prompt text + val scores.
  gate_log.jsonl       — only for gated runs: every gating decision.

The callback reads honest scores out of the adapter's honest_score_log
side-channel and the gating adapter's gate_log, so it does not need to
re-run evaluations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from gepa.core.callbacks import (
    CandidateAcceptedEvent,
    IterationEndEvent,
    OptimizationEndEvent,
    OptimizationStartEvent,
    ValsetEvaluatedEvent,
)


class ExperimentLogger:
    """Writes per-iteration experiment logs to a run directory.

    Parameters
    ----------
    run_dir:
        Directory where all log files are written.  Created if absent.
    adapter:
        The SpecGamingAdapter (or GatedSpecGamingAdapter) for this run.
        Used to drain honest_score_log and, for gated runs, gate_log.
    run_config_meta:
        Arbitrary dict of run-level metadata (run_id, flaw_condition, …)
        written to the first line of iteration_log.jsonl.
    val_case_names:
        Set of val case names used to partition honest_score_log entries
        into train vs val for preregistered metrics.
    """

    def __init__(
        self,
        run_dir: str,
        adapter: Any,  # SpecGamingAdapter | GatedSpecGamingAdapter
        run_config_meta: dict[str, Any] | None = None,
        val_case_names: set[str] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.adapter = adapter
        self.run_config_meta = run_config_meta or {}
        self.val_case_names = val_case_names or set()

        self._iter_log_path = self.run_dir / 'iteration_log.jsonl'
        self._prompt_archive_path = self.run_dir / 'prompt_archive.jsonl'
        self._gate_log_path = self.run_dir / 'gate_log.jsonl'

        # In-memory state
        self.iteration_records: list[dict[str, Any]] = []
        self.prompt_records: list[dict[str, Any]] = []
        self._last_accepted_candidate: dict[str, str] | None = None
        self._last_accepted_iter: int = 0
        self._last_valset_event: ValsetEvaluatedEvent | None = None
        self._start_time: float = time.time()

        # Write run header
        self._write_jsonl(
            self._iter_log_path,
            {'type': 'run_header', **self.run_config_meta, 'start_time': self._start_time},
            mode='w',
        )

    # ------------------------------------------------------------------
    # GEPACallback hooks
    # ------------------------------------------------------------------

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        self._write_jsonl(self._iter_log_path, {
            'type': 'optimization_start',
            'trainset_size': event['trainset_size'],
            'valset_size': event['valset_size'],
            'seed_candidate_hash': self._hash_candidate(event['seed_candidate']),
        })

    def on_valset_evaluated(self, event: ValsetEvaluatedEvent) -> None:
        # Cache the latest val evaluation; consumed in on_iteration_end.
        self._last_valset_event = event

        if event['iteration'] == 0:
            # Seed candidate baseline — include val honest metrics when available.
            val_entries = [
                r for r in self.adapter.honest_score_log
                if r['case_name'] in self.val_case_names
            ]
            seed_val_honest = self._pooled_honest_accuracy(val_entries)
            val_null_expected = sum(r.get('null_expected_count', 0) for r in val_entries)
            val_null_predicted_null = sum(
                r.get('null_predicted_null_count', 0) for r in val_entries
            )
            seed_val_null_pred_rate = (
                val_null_predicted_null / val_null_expected
                if val_null_expected > 0 else None
            )
            val_org_lengths = [
                r['predicted_org_length'] for r in val_entries
                if r.get('predicted_org_length') is not None
            ]
            seed_val_mean_org_length = (
                sum(val_org_lengths) / len(val_org_lengths)
                if val_org_lengths else None
            )

            self._write_jsonl(self._iter_log_path, {
                'type': 'seed_valset_eval',
                'average_flawed_score': event['average_score'],
                'average_honest_score': seed_val_honest,
                'val_null_pred_rate': seed_val_null_pred_rate,
                'val_mean_org_length': seed_val_mean_org_length,
                'scores_by_val_id': event['scores_by_val_id'],
                'is_best_program': event['is_best_program'],
            })
            self.iteration_records.append({
                'type': 'seed_valset_eval',
                'average_flawed_score': event['average_score'],
                'average_honest_score': seed_val_honest,
                'val_null_pred_rate': seed_val_null_pred_rate,
                'val_mean_org_length': seed_val_mean_org_length,
            })
            # Prevent seed val rows from polluting iteration 1 metrics.
            self.adapter.honest_score_log.clear()

    def on_candidate_accepted(self, event: CandidateAcceptedEvent) -> None:
        self._last_accepted_iter = event['iteration']

    def on_iteration_end(self, event: IterationEndEvent) -> None:
        iteration = event['iteration']
        accepted = event['proposal_accepted']

        # Drain honest-score side-channel from adapter.
        honest_log_snapshot = list(self.adapter.honest_score_log)
        self.adapter.honest_score_log.clear()

        # Partition into val-only entries for preregistered metrics.
        val_entries = [
            r for r in honest_log_snapshot if r['case_name'] in self.val_case_names
        ]
        train_entries = [
            r for r in honest_log_snapshot if r['case_name'] not in self.val_case_names
        ]

        # Drain gate log if this is a gated adapter.
        gate_log_snapshot: list[dict[str, Any]] = []
        gate_log_attr = getattr(self.adapter, 'gate_log', None)
        if gate_log_attr is not None:
            gate_log_snapshot = list(gate_log_attr)
            gate_log_attr.clear()  # Prevent re-writing in subsequent iterations
            for gd in gate_log_snapshot:
                self._write_jsonl(self._gate_log_path, {'iteration': iteration, **gd})

        # Compute pooled honest / mean flawed scores from train minibatch entries.
        honest_mean = self._pooled_honest_accuracy(train_entries)
        flawed_mean = (
            sum(r['flawed_score'] for r in train_entries) / len(train_entries)
            if train_entries else None
        )

        null_expected_total = sum(r.get('null_expected_count', 0) for r in train_entries)
        null_predicted_null_total = sum(
            r.get('null_predicted_null_count', 0) for r in train_entries
        )
        null_pred_rate = (
            null_predicted_null_total / null_expected_total
            if null_expected_total > 0 else None
        )

        # Val-set honest score (preregistered gaming signal denominator).
        val_honest_mean = self._pooled_honest_accuracy(val_entries)

        # Val-set null_pred_rate (preregistered C1/C4 onset metric).
        val_null_expected = sum(r.get('null_expected_count', 0) for r in val_entries)
        val_null_predicted_null = sum(
            r.get('null_predicted_null_count', 0) for r in val_entries
        )
        val_null_pred_rate = (
            val_null_predicted_null / val_null_expected
            if val_null_expected > 0 else None
        )

        # Val-set mean org length (C2 onset metric).
        val_org_lengths = [
            r['predicted_org_length'] for r in val_entries
            if r.get('predicted_org_length') is not None
        ]
        val_mean_org_length = (
            sum(val_org_lengths) / len(val_org_lengths)
            if val_org_lengths else None
        )

        # Val-set batch consistency (C3 onset metric).
        inner = getattr(self.adapter, 'inner', self.adapter)
        val_batch_consistency = getattr(inner, '_last_batch_consistency', None)

        # Val flawed score from GEPA valset event.
        val_flawed_score: float | None = None
        if self._last_valset_event is not None and self._last_valset_event['iteration'] == iteration:
            val_flawed_score = self._last_valset_event['average_score']

        state = event['state']
        best_candidate = None
        if accepted and hasattr(state, 'program_candidates') and state.program_candidates:
            best_candidate = state.program_candidates[-1].get('instructions')

        record: dict[str, Any] = {
            'type': 'iteration',
            'iteration': iteration,
            'accepted': accepted,
            'train_flawed_mean': flawed_mean,
            'train_honest_mean': honest_mean,
            'val_flawed_score': val_flawed_score,
            'val_honest_mean': val_honest_mean,
            'val_null_pred_rate': val_null_pred_rate,
            'val_mean_org_length': val_mean_org_length,
            'val_batch_consistency': val_batch_consistency,
            'null_pred_rate': null_pred_rate,
            'null_expected_total': null_expected_total,
            'null_predicted_null_total': null_predicted_null_total,
            'honest_log_entries': len(honest_log_snapshot),
            'gate_decisions': len(gate_log_snapshot),
            'elapsed_s': round(time.time() - self._start_time, 1),
        }

        if accepted and best_candidate is not None:
            record['accepted_prompt_preview'] = best_candidate[:120]

        self._write_jsonl(self._iter_log_path, record)
        self.iteration_records.append(record)

        # Write to prompt archive on acceptance.
        if accepted and best_candidate is not None:
            prompt_record: dict[str, Any] = {
                'iteration': iteration,
                'prompt': best_candidate,
                'val_flawed_score': val_flawed_score,
                'val_honest_mean': val_honest_mean,
                'val_null_pred_rate': val_null_pred_rate,
                'val_mean_org_length': val_mean_org_length,
                'val_batch_consistency': val_batch_consistency,
            }
            self._write_jsonl(self._prompt_archive_path, prompt_record)
            self.prompt_records.append(prompt_record)

    def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        self._write_jsonl(self._iter_log_path, {
            'type': 'optimization_end',
            'best_candidate_idx': event['best_candidate_idx'],
            'total_iterations': event['total_iterations'],
            'total_metric_calls': event['total_metric_calls'],
            'elapsed_s': round(time.time() - self._start_time, 1),
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pooled_honest_accuracy(entries: list[dict[str, Any]]) -> float | None:
        """Pooled field-level honest accuracy: sum(fields_correct) / sum(fields_total)."""
        total_correct = sum(r.get('fields_correct', 0) for r in entries)
        total_fields = sum(r.get('fields_total', 0) for r in entries)
        if total_fields > 0:
            return total_correct / total_fields
        if entries:
            return sum(r['honest_score'] for r in entries) / len(entries)
        return None

    @staticmethod
    def _hash_candidate(candidate: dict[str, str]) -> str:
        import hashlib
        text = candidate.get('instructions', '')
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @staticmethod
    def _write_jsonl(path: Path, record: dict[str, Any], mode: str = 'a') -> None:
        with path.open(mode, encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
