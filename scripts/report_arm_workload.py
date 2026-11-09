"""Record per-stage and per-condition PPO comparisons without selecting from test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from myo.train_vertical_ppo import write_json


def report(root):
    summaries={p.parent.name:json.loads(p.read_text()) for p in sorted(root.glob('*/summary.json'))}
    write_json(root/'comparison.json',summaries)
    lines=['# Нагрузки, ориентации и затраты: журнал PPO','','| Ветка | Новых transitions | Причина остановки | Best val, ° | Pass | Mean a² | Active fraction |',
           '| --- | ---: | --- | ---: | --- | ---: | ---: |']
    for name,s in summaries.items():
        v=s['best_validation']
        lines.append(f"| {name} | {s.get('added_transitions',s['history'][-1]['added_transitions'])} | "
          f"{s.get('stop_reason',s['status'])} | {v['score_deg']:.3f} | {v['passed']}/{v['total']} | "
          f"{v['activation_squared_mean']:.5f} | {v['active_fraction_mean']:.5f} |")
    for name,s in summaries.items():
        lines+=['',f'## {name}','','| Новых transitions | Mean tail, ° | Pass | Failures | Mean a² | Active fraction | SGD |',
                '| ---: | ---: | --- | ---: | ---: | ---: | ---: |']
        for h in s['history']:
            v=h['validation']
            lines.append(f"| {h['added_transitions']} | {v['score_deg']:.3f} | {v['passed']}/{v['total']} | "
                f"{v['failures']} | {v['activation_squared_mean']:.5f} | {v['active_fraction_mean']:.5f} | "
                f"{h['optimizer_steps']-s['source_optimizer_steps']} |")
        lines+=['','### Best: группы','','| Параметр | Значение | Ошибка, ° | Pass |','| --- | --- | ---: | --- |']
        for field,groups in s['best_validation']['groups'].items():
            for value,g in groups.items():
                lines.append(f"| {field} | {value} | {g['score_deg']:.3f} | {g['passed']}/{g['total']} |")
        for label in ('initial_test','best_test','last_test','energy_best_test',
                      'best_long_holds','energy_best_long_holds'):
            if label in s:
                v=s[label]
                lines+=['',f"{label}: {v['score_deg']:.3f}°; {v['passed']}/{v['total']}; failures{v['failures']}."]
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries={p.parent.name:json.loads(p.read_text()) for p in sorted(root.glob('*/summary.json'))}
    groups={}
    for name,s in summaries.items():
        key=json.dumps(s['training']['validation'],sort_keys=True)
        groups.setdefault(key,[]).append((name,s))
    fig,axes=plt.subplots(len(groups),3,figsize=(14,4*len(groups)),
                          squeeze=False,constrained_layout=True)
    for row,series in zip(axes,groups.values(),strict=True):
        for name,s in series:
            x=[r['added_transitions'] for r in s['history']]
            for axis,field in zip(row,('score_deg','activation_squared_mean','active_fraction_mean'),strict=True):
                axis.plot(x,[r['validation'][field] for r in s['history']],marker='.',label=name)
        for axis,title in zip(row,('Validation mean tail RMSE (deg)','Mean activation squared','Mean active fraction proxy'),strict=True):
            axis.set(xlabel='Additional PPO transitions',title=title)
            axis.grid(alpha=.25);axis.legend(fontsize=7)
    fig.suptitle('Each row uses the same validation conditions; different rows are separate tasks')
    for ext in ('png','svg'):fig.savefig(root/('comparison.'+ext),dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('runs/arm_workload'))
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    report(args.root)
    if args.plot:plot(args.root)


if __name__=='__main__':main()
