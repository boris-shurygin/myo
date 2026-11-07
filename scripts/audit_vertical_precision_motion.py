"""Replay frozen test actions at 1 ms and quantify overshoot and braking."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from myo.train_vertical_ppo import write_json
from myo.vertical_muscles import VerticalConfig, VerticalMuscles


def replay(case, config):
    physical = VerticalMuscles(config)
    physical.reset(case["initial_position_m"])
    samples = [[0., physical.position, physical.velocity]]
    original_step = mujoco.mj_step

    def record(model, data):
        original_step(model, data)
        samples.append([physical.time, physical.position, physical.velocity])

    with patch("myo.vertical_muscles.mujoco.mj_step", side_effect=record):
        for t, q, v, target, upper, lower in case["samples"]:
            physical.step([upper, lower])
            np.testing.assert_array_equal([physical.time, physical.position, physical.velocity], [t, q, v])
    return np.asarray(samples)


def metrics(a, initial, target):
    t, q, v = a.T
    direction = np.sign(target-initial)
    progress_speed = direction*v
    remaining = direction*(target-q)
    peak = int(np.argmax(progress_speed))
    inside = (np.abs(q-target) <= .005) & (np.abs(v) <= .03)
    stays_inside = np.logical_and.accumulate(inside[::-1])[::-1]
    settled = np.flatnonzero(stays_inside)
    close = np.flatnonzero(remaining <= .01)
    tail = t >= 3.
    return {"initial_position_m": initial, "target_m": target,
        "overshoot_m": max(0., float(np.max(-remaining))),
        "max_reverse_speed_m_s": max(0., float(-np.min(progress_speed))),
        "max_backtrack_m": float(np.max(np.maximum.accumulate(direction*(q-initial))-direction*(q-initial))),
        "peak_speed_m_s": float(progress_speed[peak]), "peak_speed_time_s": float(t[peak]),
        "distance_to_target_at_peak_speed_m": float(remaining[peak]),
        "speed_at_first_10mm_remaining_m_s": float(progress_speed[close[0]]) if len(close) else None,
        "settling_time_s": float(t[settled[0]]) if len(settled) else None,
        "settling_definition": "abs(position error)<=5mm and abs(velocity)<=.03m/s for ALL remaining samples until 4s",
        "last_second_velocity_peak_to_peak_m_s": float(np.ptp(v[tail])),
        "last_second_position_rmse_m": float(np.sqrt(np.mean((q[tail]-target)**2)))}


def main():
    root = Path("runs/vertical_precision/combined")
    config = VerticalConfig(**json.loads((root/"best.zip.metadata.json").read_text())["contract"]["physical"]["config"])
    assert config.physics_dt == .001
    traces, results = {}, {}
    for name in ("baseline", "best"):
        cases = json.loads((root/f"{name}_test.json").read_text())["cases"]
        traces[name], results[name] = [], []
        for case in cases:
            if case["initial_position_m"] == case["target_m"]:
                continue
            a = replay(case, config)
            traces[name].append(a)
            results[name].append(metrics(a, case["initial_position_m"], case["target_m"]))
    write_json(root/"motion_audit.json", {"physics_dt_s": config.physics_dt, "control_dt_s": config.dt,
        "all_control_boundaries_exactly_reproduced": True, "new_training": False, "results": results})
    lines = ["## Движение, перелёт и торможение", "",
        "Дополнительный read-only аудит: сохранённые test-команды повторены в native-физике.",
        "На каждом физическом подшаге 1 ms записаны q/v; все границы control-шагов 20 ms",
        "побитово совпали с исходными траекториями. Нового обучения или отбора checkpoint нет.", "",
        "Установление здесь строже исходного pass: ошибка ≤5 mm и скорость ≤0,03 m/s",
        "на ВСЕХ оставшихся отсчётах до конца 4 s. Это не гарантия удержания за пределами окна.",
        "Перелёт — максимальное пересечение цели в направлении первоначального движения.", "",
        "| Движение, mm | Перелёт старый→новый, mm | Пик скорости, m/s | Установление, s | Скорость за 10 mm до цели, m/s |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for old, new in zip(results["baseline"], results["best"], strict=True):
        pair = lambda key, scale=1., digits=3: f"{old[key]*scale:.{digits}f} → {new[key]*scale:.{digits}f}"
        lines.append(f"| {new['initial_position_m']*1000:+.0f}→{new['target_m']*1000:+.0f} | {pair('overshoot_m',1000,2)} | {pair('peak_speed_m_s')} | {pair('settling_time_s')} | {pair('speed_at_first_10mm_remaining_m_s')} |")
    lines.extend(["", "На всех четырёх движениях новая модель не пересекает цель даже на сетке 1 ms.",
        "Она снижает пиковую скорость и приходит к цели с меньшей скоростью, но медленнее",
        "устанавливается. Нулевой перелёт отчасти сочетается с недоходом: +60 mm достигается",
        "лишь до примерно +55,34 mm; это не точное попадание в центр допуска.",
        f"Пик скорости подъёма −30→+60 сместился с {results['baseline'][0]['peak_speed_time_s']*1000:.0f} до {results['best'][0]['peak_speed_time_s']*1000:.0f} ms, при оставшемся расстоянии",
        f"{results['best'][0]['distance_to_target_at_peak_speed_m']*1000:.1f} mm у новой модели. Это наблюдаемое замедление, а не доказательство",
        "внутренней модели задержек или отдельного планирования тормозного пути.",
        f"Торможение вниз не идеально плавное: новая модель сначала притормаживает до цели,",
        f"ненадолго движется обратно (откаты {results['best'][1]['max_backtrack_m']*1000:.2f} и {results['best'][3]['max_backtrack_m']*1000:.2f} mm), затем снова приближается к цели.",
        "Отсутствие перелёта не означает отсутствия колебаний или ранних реверсов.",
        "График показывает первые 1,5 s; метрики используют весь эпизод 4 s.",
        "Сценарии начинаются штатным сбалансированным reset. Реакция на смену цели во время",
        "движения, неожиданные воздействия и другие стартовые скорости этой таблицей не проверены.", ""])
    report = "\n".join(lines)
    path = Path("docs/vertical-precision-ppo.md")
    path.write_text(path.read_text().split("## Движение, перелёт и торможение")[0].rstrip()+"\n\n"+report)
    (root/"motion_report.md").write_text(report)
    print(report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True)
    for i, case in enumerate(results["best"]):
        target = case["target_m"]*1000
        axes[0, i].axhspan(target-5, target+5, color="gray", alpha=.15)
        axes[0, i].axhline(target, color="black", lw=.8)
        for name, label, style in (("baseline", "Previous", "--"), ("best", "Dense/log", "-")):
            t, q, v = traces[name][i].T
            axes[0, i].plot(t, q*1000, style, label=label)
            axes[1, i].plot(t, v, style)
        axes[0, i].set_title(f"{case['initial_position_m']*1000:+.0f} to {target:+.0f} mm")
        axes[0, i].set_ylabel("Position, mm")
        axes[1, i].set(xlabel="Time, s", ylabel="Velocity, m/s", xlim=(0, 1.5))
        axes[1, i].axhline(0, color="black", lw=.8)
        for ax in axes[:, i]:
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(root/f"motion_comparison.{ext}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
