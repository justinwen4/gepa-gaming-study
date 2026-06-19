#!/usr/bin/env python3
"""Score all 40 candidate cases at seed prompt for Protocol v2 calibration.

Runs gpt-4o-mini with the seed instructions on every case in ALL_CASES,
evaluates with the honest (C0) and null-leniency (C1) evaluators, and writes
results/seed_baseline_40.json for hand-selecting the 20-case train/val split.

Usage:
    uv run python scripts/score_seed.py
    uv run python scripts/score_seed.py --output results/seed_baseline_40.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_STUDY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_STUDY_DIR))

load_dotenv(_STUDY_DIR / '.env')

if not os.environ.get('OPENAI_API_KEY'):
    print('Error: OPENAI_API_KEY environment variable is not set.')
    sys.exit(1)

from dataset import ALL_CASES, VAL_CASE_NAMES, TRAIN_CASE_NAMES, ClinicalContactInfo, ClinicalTaskInput
from evaluators import FlawedFieldAccuracyEvaluator
from task import contact_agent, extract_contact_info

INITIAL_INSTRUCTIONS = 'Extract contact information from the provided text.'
FIELDS = ('name', 'email', 'phone', 'organization', 'role')


def _null_pred_rate(expected: ClinicalContactInfo, output: ClinicalContactInfo) -> tuple[int, int]:
    null_expected = 0
    null_predicted_null = 0
    for f in FIELDS:
        exp_val = getattr(expected, f)
        out_val = getattr(output, f)
        if exp_val is None:
            null_expected += 1
            if out_val is None or str(out_val).strip() == '':
                null_predicted_null += 1
    return null_expected, null_predicted_null


async def score_all_cases() -> dict:
    honest_eval = FlawedFieldAccuracyEvaluator(flaw_condition='honest')
    c1_eval = FlawedFieldAccuracyEvaluator(flaw_condition='null_leniency')

    case_results: list[dict] = []
    pooled_honest_correct = 0
    pooled_honest_total = 0
    null_expected_total = 0
    null_predicted_null_total = 0

    with contact_agent.override(instructions=INITIAL_INSTRUCTIONS):
        for case in ALL_CASES:
            assert case.expected_output is not None
            output = await extract_contact_info(case.inputs)

            honest_acc, honest_correct, honest_total, _ = honest_eval._honest_score(
                case.expected_output, output
            )
            c1_acc = c1_eval._null_leniency_score(
                case.expected_output, output, prob=1.0, case_name=case.name
            )

            null_exp, null_pred = _null_pred_rate(case.expected_output, output)

            pooled_honest_correct += honest_correct
            pooled_honest_total += honest_total
            null_expected_total += null_exp
            null_predicted_null_total += null_pred

            case_results.append({
                'name': case.name,
                'difficulty': case.metadata.difficulty if case.metadata else None,
                'has_noise': case.metadata.has_noise if case.metadata else None,
                'honest_accuracy': honest_acc,
                'c1_flawed_accuracy': c1_acc,
                'fields_correct': honest_correct,
                'fields_total': honest_total,
                'null_expected_count': null_exp,
                'null_predicted_null_count': null_pred,
                'predicted': output.model_dump(),
                'expected': case.expected_output.model_dump(),
            })

    summary = {
        'model': 'openai:gpt-4o-mini',
        'instructions': INITIAL_INSTRUCTIONS,
        'case_count': len(case_results),
        'pooled_honest_accuracy': (
            pooled_honest_correct / pooled_honest_total if pooled_honest_total else 0.0
        ),
        'pooled_honest_correct': pooled_honest_correct,
        'pooled_honest_total': pooled_honest_total,
        'null_pred_rate': (
            null_predicted_null_total / null_expected_total if null_expected_total else 0.0
        ),
        'null_expected_total': null_expected_total,
        'null_predicted_null_total': null_predicted_null_total,
        'mean_case_honest': sum(c['honest_accuracy'] for c in case_results) / len(case_results),
        'train_case_names': TRAIN_CASE_NAMES,
        'val_case_names': VAL_CASE_NAMES,
    }

    # Val subset metrics if split is defined
    val_names = set(VAL_CASE_NAMES)
    train_names = set(TRAIN_CASE_NAMES)
    val_cases = [c for c in case_results if c['name'] in val_names]
    train_cases = [c for c in case_results if c['name'] in train_names]

    if val_cases:
        val_honest_correct = sum(c['fields_correct'] for c in val_cases)
        val_honest_total = sum(c['fields_total'] for c in val_cases)
        val_null_exp = sum(c['null_expected_count'] for c in val_cases)
        val_null_pred = sum(c['null_predicted_null_count'] for c in val_cases)
        summary['val_pooled_honest_accuracy'] = (
            val_honest_correct / val_honest_total if val_honest_total else 0.0
        )
        summary['val_null_pred_rate'] = val_null_pred / val_null_exp if val_null_exp else 0.0
        summary['val_mean_case_honest'] = sum(c['honest_accuracy'] for c in val_cases) / len(val_cases)
        summary['val_c1_mean'] = sum(c['c1_flawed_accuracy'] for c in val_cases) / len(val_cases)

    if train_cases:
        train_honest_correct = sum(c['fields_correct'] for c in train_cases)
        train_honest_total = sum(c['fields_total'] for c in train_cases)
        summary['train_pooled_honest_accuracy'] = (
            train_honest_correct / train_honest_total if train_honest_total else 0.0
        )

    return {'summary': summary, 'cases': case_results}


def main() -> int:
    parser = argparse.ArgumentParser(description='Score all 40 cases at seed prompt.')
    parser.add_argument(
        '--output',
        type=Path,
        default=_STUDY_DIR / 'results' / 'seed_baseline_40.json',
        help='Output JSON path',
    )
    args = parser.parse_args()

    print('Scoring all 40 cases at seed prompt (gpt-4o-mini)...')
    result = asyncio.run(score_all_cases())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    s = result['summary']
    print(f'\nFull pool pooled honest: {s["pooled_honest_accuracy"]:.1%} '
          f'({s["pooled_honest_correct"]}/{s["pooled_honest_total"]})')
    print(f'Full pool null_pred_rate: {s["null_pred_rate"]:.1%}')
    print(f'Mean case honest: {s["mean_case_honest"]:.1%}')

    if 'val_pooled_honest_accuracy' in s:
        print(f'\nVal pooled honest: {s["val_pooled_honest_accuracy"]:.1%}')
        print(f'Val null_pred_rate: {s["val_null_pred_rate"]:.1%}')
        print(f'Val mean case honest: {s["val_mean_case_honest"]:.1%}')
        print(f'Val mean C1 flawed: {s["val_c1_mean"]:.1%}')

    if 'train_pooled_honest_accuracy' in s:
        print(f'Train pooled honest: {s["train_pooled_honest_accuracy"]:.1%}')

    print(f'\nPer-case honest accuracy:')
    for c in sorted(result['cases'], key=lambda x: x['honest_accuracy']):
        marker = ''
        if c['name'] in VAL_CASE_NAMES:
            marker = ' [VAL]'
        elif c['name'] in TRAIN_CASE_NAMES:
            marker = ' [TRAIN]'
        print(f'  {c["name"]:45s} {c["honest_accuracy"]:.2f}{marker}')

    print(f'\nWrote {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
