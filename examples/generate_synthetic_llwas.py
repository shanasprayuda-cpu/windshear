#!/usr/bin/env python3
"""Create small, fictional minute-level LLWAS-style data; NOT a research dataset.

No real station measurements, true windshear events or real locations are used.
This CSV demonstrates column naming only. It is NOT compatible with canonical
Stage C manifests and MUST NOT be used to reproduce paper metrics.
"""
import argparse
import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

SENSORS = [f'A{i}' for i in range(2, 7)] + [f'B{i}' for i in range(2, 7)]


def generate(path: Path, minutes: int) -> int:
    if minutes < 80:
        raise ValueError('Use at least 80 minutes to illustrate a 60+10 timeline')
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ['timestamp', 'timestamp_present', 'label', 'label_valid']
    for s in SENSORS:
        cols.extend([f'{s}_u_ms', f'{s}_v_ms'])
    cols.extend(f'{s}_imputed' for s in SENSORS)
    cols += ['any_imputed', 'all_sensors_available']
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for t in range(minutes):
            row = {
                'timestamp': (base + timedelta(minutes=t)).strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp_present': 1, 'label': int(t % 47 == 0),
                'label_valid': 1, 'any_imputed': 0,
                'all_sensors_available': 1,
            }
            for j, s in enumerate(SENSORS):
                phase = t * 0.05 + j * 0.2
                row[f'{s}_u_ms'] = f'{2.0 * math.sin(phase):.5f}'
                row[f'{s}_v_ms'] = f'{1.5 * math.cos(phase):.5f}'
                row[f'{s}_imputed'] = 0
            writer.writerow(row)
    return minutes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('examples/synthetic_llwas_minute.csv'))
    parser.add_argument('--minutes', type=int, default=120)
    args = parser.parse_args()
    print(f'Wrote {generate(args.out, args.minutes)} fictional minute records to {args.out}')
    print('STRUCTURAL DEMO ONLY — not valid for model training or published metrics.')


if __name__ == '__main__':
    main()
