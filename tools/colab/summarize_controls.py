"""Summarize a verified, complete frozen-control campaign without tuning it."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(report):
    if report.get('status') != 'complete' or report.get('paired_seeds') != [7, 42, 123]:
        raise ValueError('A complete three-seed comparison is required')
    variants = report['models']
    expected = {'symmetric_single', 'symmetric_sequential', 'asymmetric', 'symmetric_matched'}
    if set(variants) != expected:
        raise ValueError('All four frozen comparison arms are required')
    rows = {entry['seed']: entry['models'] for entry in report['runs']}
    if set(rows) != {7, 42, 123}:
        raise ValueError('Missing or duplicate seed results')
    for pair in rows.values():
        for variant in variants:
            if pair[variant]['epochs_completed'] != 12:
                raise ValueError('Incomplete fixed training horizon')
    scores = {variant: [rows[seed][variant]['full_val_metrics']['val_vqa_acc']
                        for seed in (7, 42, 123)] for variant in variants}
    lines = ['# Frozen VQA shared-control results', '',
             'Full validation VQA accuracy after selecting each checkpoint on the fixed validation subset.', '',
             '| Architecture | Seed 7 | Seed 42 | Seed 123 | Mean | Sample SD |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for variant, values in scores.items():
        formatted = ' | '.join(f'{value:.4f}' for value in values)
        lines.append(f'| {variant} | {formatted} | {statistics.mean(values):.4f} | {statistics.stdev(values):.4f} |')
    lines += ['', 'Mean paired asymmetric-minus-control differences, in percentage points:', '']
    for variant in variants:
        if variant == 'asymmetric':
            continue
        differences = [a - b for a, b in zip(scores['asymmetric'], scores[variant])]
        lines.append(f'- Versus {variant}: {statistics.mean(differences):+.4f}; '
                     f'paired differences {", ".join(f"{value:+.4f}" for value in differences)}.')
    lines += ['', 'These are three paired training seeds. The single shared block changes capacity; '
              'the sequential shared control changes depth and aggregation. The results do not '
              'independently establish directional specialization or modality roles as a mechanism.',
              '', 'Latency/memory/operator coverage are in the separate profile JSON. Historical '
              'A100 results must remain separate from the new L4 measurements.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', required=True, type=Path)
    args = parser.parse_args()
    paths = list((args.campaign_dir / 'monitor/artifacts').rglob('*_final_training_report.json'))
    if len(paths) != 1:
        raise ValueError(f'Expected one final verified report; found {len(paths)}')
    text = summarize(json.loads(paths[0].read_text()))
    target = args.campaign_dir / 'results_summary.md'
    target.write_text(text)
    print(target)


if __name__ == '__main__':
    main()
