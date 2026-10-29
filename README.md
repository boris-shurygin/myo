# myo — физические персонажи с мышечным управлением

Хобби-проект: персонаж получает команды, нейросетевой контроллер возбуждает
мышцы, физика рассчитывает движение, а визуализация показывает деформацию мышц
и их усилия. Обучение может работать без графики; конечная цель — управление и
отрисовка в реальном времени. После проверки пяти демо — решение о переходе к
симулятору баскетбольной тактики.

## Текущее состояние: V0

Это первая исполняемая экспериментальная площадка для **изолированной руки на
скамье Скотта**. Плечо закреплено, локоть имеет одну степень свободы, груз движется
вместе с предплечьем. Четыре условные мышцы: `biceps_long`, `biceps_short`,
`brachialis`, `triceps`.

| Направление | Уже в коде | Следующая проверка |
|---|---|---|
| Мышечный контроллер | Gymnasium + PPO; аналитический эталон сгибания; diagnostic heuristic | Достаточное обучение и оценка; затем MyoSuite/MuscleMimic для сложного тела |
| Задержки | Явная очередь моторных команд и отдельная динамика активации | Рандомизация задержки, сравнение способов компенсации |
| Упругие сухожилия | Отдельная длина волокна, tension-only tendon, strain и упругая энергия | Калибровка и переход к более подробной модели |
| Нагрузка и утомление | Capacity proxy; отдельные ROM/load/fiber/tendon penalties | Абляции, пространственные пути мышц и проверка нагрузок |
| Предиктор | Сбор эпизодов, NumPy MLP, frozen runtime wrapper | Польза для learned policy против истории/большего actor |
| Отображение усилий | Телеметрия activation/force/strain/capacity, локальный UDP sender, обычный MuJoCo viewer | Адаптер UE, активная деформация Flesh и обучение ML Deformer |

**V0 не воспроизводит MuscleMimic, не является валидированной моделью человека и
не содержит готового UE-проекта.** Полноценная мышечная геометрия ещё не реализована:
MuJoCo viewer отображает исходную механическую сцену. Короткие проверки обучения
подтверждают работоспособность pipeline, а не освоение движения.

Фактически выполненные проверки и ограничения: [docs/validation.md](docs/validation.md).

В начальной проверке прошли **76 тестов**. После **100096 переходов** PPO получил
RMSE угла **6.97–8.38°** на трёх проверочных эпизодах, без физического срыва;
heuristic на тех же эпизодах даёт **2.79–3.30°**. Это рабочий learned baseline,
но качество первого демо и польза предиктора для управления ещё требуют оценки.

## Быстрый запуск

Проверяемое базовое окружение — Python 3.12 на Linux, без GPU и без дисплея для
headless режима. Основные зависимости: MuJoCo 3.17, Gymnasium 1.4 и NumPy 2.
Точные проверенные версии приведены в `requirements-verified.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'

python3 -m pytest -q
python3 -m myo rollout --seconds 10 --controller heuristic
python3 -m myo rollout --seconds 10 --controller zero
```

Для обычного окна MuJoCo нужен рабочий графический дисплей:

```bash
python3 -m myo rollout --seconds 20 --viewer --realtime
```

`--realtime` ограничивает темп по часам и не изменяет шаг физики. Без этого флага
симуляция идёт настолько быстро, насколько позволяет машина. В viewer используется
диагностический контроллер, пока явно не передан путь к обученной policy.

Полная конфигурация первого демо: [configs/preacher_curl.json](configs/preacher_curl.json).

```bash
python3 -m myo rollout --config configs/preacher_curl.json \
  --seconds 10 --output runs/curl.jsonl
```

## Обучение контроллера

PyTorch и Stable-Baselines3 нужны только для PPO. На Linux/Windows для CPU можно
сначала установить CPU-сборку PyTorch из официального индекса:

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -e '.[train]'

python3 -m myo.train --steps 100000 --seed 0 --threads 1 \
  --output runs/ppo_curl.zip

python3 -m myo rollout --controller runs/ppo_curl.zip --seconds 10
```

Число шагов — бюджет запуска, не обещание сходимости. PPO округляет бюджет до
полных rollout chunks; metadata сохраняет фактическое число transitions.
При запуске policy необходим sidecar `ppo_curl.zip.metadata.json`. Он хранит
конфигурацию, observation order, версии и идентичность predictor. Несовпадение
контрактов приводит к ошибке, а не к неявной смене физики.

В V0 action — четыре возбуждения в `[0, 1]`. Они проходят через задержку,
активацию и muscle/tendon dynamics. Сеть не задаёт угол или момент сустава напрямую.
Нормализация признаков описана в `env.py`; внешнего `VecNormalize` здесь нет.

## Предиктор

Вход: текущая команда задачи (целевые угол и скорость), состояние и доступная
история из base observation. Новая кандидатная команда возбуждения мышцы во вход не входит.
Выход: две знаковые ошибки через 100 ms — угла и скорости относительно команды.
Во время исполнения это один проход замороженной сети, без доступа к будущим
реальным состояниям и без дополнительного прокручивания симулятора.

Проверка пути данных на диагностическом контроллере:

```bash
python3 -m myo.predictor collect --episodes 12 --horizon 0.1 \
  --controller heuristic --output runs/predictor_data.npz

python3 -m myo.predictor train --data runs/predictor_data.npz \
  --epochs 100 --output runs/predictor.json

python3 -m myo.train --steps 100000 --predictor runs/predictor.json \
  --output runs/ppo_with_predictor.zip

python3 -m myo rollout --controller runs/ppo_with_predictor.zip --seconds 10
```

Для эксперимента с обученным контроллером в `collect --controller` нужно передать
его `.zip`. Предиктор описывает поведение политики, на которой собраны данные;
после изменения контроллера его ошибку нужно измерить повторно. Данные heuristic
проверяют механизм, но не доказывают пользу predictor для PPO. Train/validation
разделены по эпизодам; текущий отчёт ещё не заменяет независимый test set.

## Визуализация и реальное время

```bash
python3 -m myo rollout --controller heuristic --seconds 20 --realtime \
  --udp-port 7001 --output runs/visualization.jsonl
```

Это односторонние снимки на `127.0.0.1:7001`; UE-получатель предстоит реализовать.
Доступны отдельные excitation, delayed excitation, activation, force в N, strain
и remaining capacity. Формат: [docs/telemetry.md](docs/telemetry.md).

План подключения: [integrations/unreal/README.md](integrations/unreal/README.md).
MuJoCo остаётся владельцем физики; графика получает согласованное состояние.
Flesh/ML Deformer не изменяют физику через неявную обратную связь.

## Последовательность демо

1. **Рука на скамье Скотта:** отработать всю цепочку от мышечной механики до UE.
2. **Присед со своим весом:** баланс, естественность и нагрузочные сигналы.
3. **Ходьба по командам:** скорость, направление, старт и остановка.
4. **Бег со сменой направления:** ускорения, торможения и переходы.
5. **Защитные перемещения:** одновременно двигаться и отслеживать мяч/объекты,
   проверять совмещение задач и выбор паттернов.

Подробные этапы, критерии выхода и границы проекта: [docs/roadmap.md](docs/roadmap.md).

## Карта репозитория

| Файл / каталог | Назначение |
|---|---|
| `src/myo/delay.py`, `muscle.py` | Причинная задержка и упругая мышечно-сухожильная модель |
| `src/myo/env.py`, `assets/` | MuJoCo-сцена, observation/action/reward и baseline |
| `src/myo/train.py` | Обучение PPO и metadata артефактов |
| `src/myo/predictor.py` | Датасет, обучение предиктора, runtime wrapper |
| `src/myo/runtime.py`, `cli.py`, `telemetry.py` | Проверка checkpoint contract, запуск, JSONL/UDP |
| `tests/` | Проверки динамики, причинности, данных и контрактов |
| [architecture.md](docs/architecture.md) | Выбранные решения и их границы |
| [model.md](docs/model.md) | Уравнения и ограничения модели V0 |
| [experiments.md](docs/experiments.md) | Абляции, метрики и защита от утечки будущего |
| [runtime.md](docs/runtime.md) | CPU/GPU, расписание и измерения реального времени |
| [research.md](docs/research.md) | Первичные источники и отличие от готовых фреймворков |
| [licenses.md](docs/licenses.md) | Код отдельно от моделей, данных и checkpoints |

## Лицензии и данные

Лицензия собственного кода проекта пока не выбрана владельцем. Исходная модель
и аналитический эталон созданы для этого репозитория; здесь нет AMASS, SMPL,
MoBL-ARMS, чужих checkpoints или анатомических мешей. Лицензии зависимостей и
будущих источников проверяются отдельно. Сгенерированные данные и веса по
умолчанию записываются в `runs/` и не коммитятся.

