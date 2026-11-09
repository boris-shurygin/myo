"""Compare frozen evaluations before and after extra predictor updates."""
import argparse
import json
from pathlib import Path

from myo.train_vertical_ppo import write_json


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, default=Path('runs/arm_anticipation60'))
    p.add_argument('--after', type=Path, default=Path('runs/arm_anticipation60_extended'))
    args = p.parse_args()
    old, new = read(args.before/'training.json'), read(args.after/'training.json')
    if new['status'] != 'complete':
        raise ValueError('Finish continuation and held-out evaluation first')
    a, b = read(args.before/'test.json'), read(args.after/'test.json')
    assert all(group['parent150k'] == a['groups'][name]['direct'] for name, group in b['groups'].items())
    assert old['training']['dataset_sha256'] == new['training']['dataset_sha256']
    cases_a, cases_b = read(args.before/'command_contrasts.json'), read(args.after/'command_contrasts.json')
    for left, right in zip(cases_a['cases'], cases_b['cases'], strict=True):
        for key in ('family', 'position_deg', 'step', 'actions', 'initial_physical', 'actual_delta_between_commands'):
            assert left[key] == right[key]
    policies_a, policies_b = read(args.before/'policy_probes.json'), read(args.after/'policy_probes.json')
    comparison = {'before_updates': old['best_update'], 'after_best_update': new['best_update'],
        'added_updates': new['added_updates'], 'stop_reason': new['stop_reason'],
        'same_dataset_labels_and_policy_probes': True, 'groups': {}, 'policies': {},
        'command_contrasts': {'before': cases_a['direct'], 'after': cases_b['direct']}}
    for group in a['groups']:
        comparison['groups'][group] = {'before': a['groups'][group]['direct'],
                                      'after': b['groups'][group]['direct']}
    for name in policies_a:
        assert policies_a[name]['checkpoint'] == policies_b[name]['checkpoint']
        assert policies_a[name]['persistence'] == policies_b[name]['persistence']
        comparison['policies'][name] = {'before': policies_a[name]['direct'],
                                       'after': policies_b[name]['direct']}
    write_json(args.after/'budget_comparison.json', comparison)
    lines = ['# Дополнительный бюджет предиктора60 ms', '',
        f"Best{new['best_update']}; новых updates{new['added_updates']}; stop `{new['stop_reason']}`.", '',
        '| Набор | Угол до, ° | После, ° | Скорость до, °/s | После, °/s |',
        '| --- | ---: | ---: | ---: | ---: |']
    rows = {**comparison['groups'], **comparison['policies'], 'command_contrasts': comparison['command_contrasts']}
    for name, row in rows.items():
        first, last = row['before'], row['after']
        lines.append(f"| {name} | {first['angle_rmse_deg']:.4f} | {last['angle_rmse_deg']:.4f} | "
                     f"{first['velocity_rmse_deg_s']:.3f} | {last['velocity_rmse_deg_s']:.3f} |")
    lines += ['', '| Канал | Единицы | RMSE до | После |', '| --- | --- | ---: | ---: |']
    for first, last in zip(a['groups']['all']['direct']['channels'], b['groups']['all']['direct']['channels'], strict=True):
        assert first['name'] == last['name']
        lines.append(f"| {first['name']} | {first['unit']} | {first['rmse']:.6g} | {last['rmse']:.6g} |")
    (args.after/'budget_comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'best_update': new['best_update'], 'added_updates': new['added_updates'],
        'all_before_q': comparison['groups']['all']['before']['angle_rmse_deg'],
        'all_after_q': comparison['groups']['all']['after']['angle_rmse_deg'],
        'all_before_v': comparison['groups']['all']['before']['velocity_rmse_deg_s'],
        'all_after_v': comparison['groups']['all']['after']['velocity_rmse_deg_s']}))


if __name__ == '__main__':
    main()
