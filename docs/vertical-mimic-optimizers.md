# Adam и Muon на контроллере MuscleMimic

Опыт 2026-11-06: меняем только optimizer на физическом варианте 4 — упругие
мышцы/сухожилия, motor delay 40 ms и динамика активации 15/50 ms.
Задача — imitation известных кинематических траекторий, как в
[основном опыте](vertical-musclemimic.md), а не команды ускорения.

Сохраняются actor/critic 4×256, SiLU/LayerNorm, seed=0, начальные веса и running
statistics, 128 сред, rollout 20, одна эпоха/32 minibatch, LR=0,0004,
нормализация, reward, GAE, observation/action и физика. На каждый optimizer:
128 PPO-итераций, 327 680 transitions, 4096 minibatch-обновлений.
Начальные параметры/statistics побитово совпали; initial validation также
совпадает целиком. Конфигурации и версии пакетов проверены отчётным скриптом.

Adam — исходный `build_optimizer("adamw", ..., weight_decay=0)`, эквивалентный
Adam: beta1=0,9, beta2=0,999, eps=1e−5, global grad norm ≤1.
Muon сохраняет исходные настройки: beta=0,95, Nesterov, пять Newton–Schulz
итераций, adaptive=false; weight decay=0, Adam fallback для векторов.
Для Adam отдельно LR не подбирался. Один seed не устанавливает превосходство
одного метода в общем случае.

## Точность

Best выбирается только по восьми полным validation-клипам (включая старты
с 35% клипа). Пять отложенных test-клипов оцениваются после окончания обучения.
Strict pass: полный клип, q RMSE ≤10 mm и v RMSE ≤0,05 m/s одновременно.
Средняя RMSE ниже — среднее по клипам, а не ошибка одной общей склейки.

| Optimizer | Best iteration | Best validation q, mm | Last validation q, mm | Test q, mm | Test v, m/s | Test strict | Completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Muon | 128 | 18.16 | 18.16 | 20.62 | 0.0522 | 0/5 | 5/5 |
| Adam | 48 | 30.91 | 48.67 | 29.78 | 0.0639 | 0/5 | 5/5 |

| PPO iteration | Muon validation q, mm | Adam validation q, mm |
| ---: | ---: | ---: |
| 16 | 92.72 | 92.05 |
| 32 | 64.40 | 108.07 |
| 48 | 42.33 | 30.91 |
| 64 | 47.20 | 86.08 |
| 80 | 26.56 | 108.96 |
| 96 | 21.15 | 107.10 |
| 112 | 24.93 | 90.02 |
| 128 | 18.16 | 48.67 |

| Test clip | Muon q RMSE, mm | Adam q RMSE, mm | Muon strict | Adam strict |
| --- | ---: | ---: | --- | --- |
| test_75 | 20.43 | 35.43 | нет | нет |
| test_95 | 25.26 | 43.46 | нет | нет |
| test_up | 15.72 | 29.59 | нет | нет |
| test_down | 18.87 | 26.48 | нет | нет |
| test_hold | 22.80 | 13.92 | нет | нет |

После завершения обучения дополнительно сопоставлены initial/best/last Adam
на тех же test-клипах. Эти оценки не влияли на остановку или выбор best.

| Adam checkpoint | PPO iteration | Test q RMSE, mm | Test v RMSE, m/s |
| --- | ---: | ---: | ---: |
| initial | 0 | 94.24 | 0.0995 |
| best | 48 | 29.78 | 0.0639 |
| last | 128 | 53.79 | 0.0709 |

На test есть улучшение относительно инициализации, но после validation-best
ухудшение подтверждается и на test. Это три контрольные точки, не полная
кривая test по итерациям. Полезное управление частично выучено; продолжение
до 128 итераций его не улучшило. Обе policy не прошли строгую приёмку ни
одного test-клипа. У Adam точнее test-hold, у Muon — четыре движущихся test-клипа.

![Validation, reward и отложенные движения](../runs/vertical_mimic_adam/comparison.png)

## Время

Изолированный замер: один и тот же native rollout 20×128, одинаковые начальные
веса и statistics, полный PPO loss/backprop и 32 шага optimizer. Физика, сбор
опыта и validation исключены; JAX-компиляция прогрета отдельно, ожидание результата
явное. Шесть замеров каждого optimizer чередуются на тех же двух CPU threads.

| Optimizer | Медиана полного PPO-update после прогрева, s |
| --- | ---: |
| Muon | 2.2897 |
| Adam | 0.1961 |

В этом замере PPO-update с Muon дороже в **11.68 раза**.
Это стоимость всей оптимизации с backprop, не отдельно Muon/Adam kernel.

Во время замера native-аудит работал на других CPU cores; частоты и
тепловой режим не фиксировались. Все шесть исходных времён каждого optimizer
сохранены в `update_benchmark.json`; отношение — ориентир для этого CPU.

- Muon: весь опыт 1572.8 s; медиана rollout+update без первой итерации 10.346 s.
- Adam: весь опыт 393.7 s; медиана rollout+update без первой итерации 2.406 s.

У Adam медиана сбора опыта (forward, native физика, обмен NumPy/JAX) — 2.264 s, PPO-update — 0.144 s.
Muon — прежний опыт с четырьмя одновременно обучаемыми вариантами; Adam
запускался отдельно (часть времени параллельно шли тесты на других CPU cores).
Поэтому отношение полного wall time не является изолированной оценкой optimizer.

## Воспроизведение

```bash
.venv/bin/python3 -m myo.train_vertical_mimic --variant 4 --optimizer adam --output-dir runs/vertical_mimic_adam/variant_4 --num-envs 128 --updates 128 --hidden 256 256 256 256 --eval-every 16 --seed 0 --cpu-threads 2
.venv/bin/python3 scripts/evaluate_vertical_mimic.py --root runs/vertical_mimic_adam --variants 4 --checkpoint-progress
.venv/bin/python3 scripts/benchmark_vertical_mimic_optimizers.py
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_vertical_mimic_optimizers.py
```

Checkpoint сохраняет выбор и полное состояние optimizer. Совместимость исходных
Muon-checkpoints сохранена по явно закреплённому прежнему hash trainer; остальные
проверки исходников/физики/наблюдений/весов остаются обязательными.
Веса, benchmark и траектории остаются в ignored `runs/`. Новое GUI не проверялось.
