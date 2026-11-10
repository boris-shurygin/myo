# Проверки проекта

## Подготовка curriculum экономии усилий — 2026-11-10

```bash
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m pytest -q \
  tests/test_arm_exploration.py tests/test_arm_reward_curriculum.py
```

**6 passed in26,00 s**. Проверены прежние физические/likelihood/roundtrip
инварианты и защита effort-selection от дешёвого провала цели или превышения
допустимой ошибки. Используется прежний `position_efficiency` wrapper;
runner задаёт коэффициенты явно и сохраняет их с новым training snapshot.
Изменённые веса награды не допускают переноса старой plateau-reference.
Исходный успешный checkpoint и pinned physical/reward modules не изменены.
Протокол —[arm-effort-curriculum.md](arm-effort-curriculum.md).

Полный запуск выполнен с теми же CPU/priority/BLAS ограничениями вне socket
sandbox,чтобы включить локальный UDP-loopback:

```bash
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m pytest -q
```

**192 passed,6 warnings in95,50 s**. Это единый полный запуск,UDP не исключён.
Warnings —существующие Gymnasium observation-space предупреждения.

Четыре последовательных training jobs завершены:weight0/0,1 при LR1e−5,
по65536 transitions;weight0/0,1 при LR3e−6,по32768. Всего196608 transitions.
Best всех четырёх остался initial:новой policy,сохранившей все18 train и27
validation pass со снижением затрат,нет. Last меньшего LR со штрафом
дешевле last контроля на4,398% по равновзвешенному mean(a²) train/validation,
но оба проходят16/18 и25/27. Это не успех экономизации и не доказанное плато.
Полные таблицы/выбор —в журнале опыта,данные —
`runs/arm_effort_curriculum/comparison.json`.

Начальные policy/optimizer serialized payloads первого matched-опыта
побитово совпали с parent;первые1024 исходных учебных перехода совпали
по физике/командам/mean/logp. Разница reward равна ожидаемому штрафу;
max absolute error1,70e−17. Для выбранных best проверены checkpoint SHA,
совпадение policy payloads и физического/input/detector contract с parent
после удаления только reward mode/коэффициентов. Старый test8/8;0,670°
переиспользован только для неизменившегося parent,не для новых last.

Отдельно с тем же resource-prefix:

```bash
.venv/bin/python3 scripts/diagnose_arm_effort.py
```

Выполнены15 native replay(5 policy×3 удержания),по200 шагов,без physical
failures. Диагностика разделила position/speed fractions:для Control LR1e−5
Hold135°,tilt150°,2 кг position fraction1,0,speed fraction0,333,RMS скорости
7,665°/s против0,779°/s у parent. Для Activation0.1 LR1e−5 Hold135°,tilt120°
ошибка5,363° при RMS скорости0,254°/s. Нельзя считать эти разные провалы
одной лишь нехваткой активации или одним лишь дрожанием.
Артефакт —`failure_diagnostics.json`,включая per-step forces/activation/commands.
Наблюдаемое несовпадение position-only reward и speed-критерия pass
зафиксировано как направление следующей проверки,не как установленная
единственная причина неуспеха PPO. Все свои training/diagnostic/test jobs
завершены с exit0;новых работающих процессов обучения не оставлено.

## Продолжение PPO с журналом эпизодов — 2026-11-10

```bash
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m pytest -q tests/test_arm_exploration.py
```

Результат: **4 passed in15,95 s**. Новый реальный PPO-сценарий с двумя worker
и двумя rollout проверил512 записанных переходов и два завершённых эпизода,
пересекающих update. Веса с включённым и выключенным audit побитово совпали;
сырые advantages сопоставлены с исходными time/worker после SB3 flatten.
Проверены полный train+validation gate,прежний detector/контракт/roundtrip.
Это проверка диагностического кода,а не успеха дальнейшего обучения.
Изменены runner и отдельный read-only audit;физические модули,наблюдения,
награда,action mapping и `arm_exploration.py` остаются прежними.

Дополнительный запуск с тем же resource-prefix:

```bash
.venv/bin/python3 -m pytest -q \
  --deselect=tests/test_telemetry.py::test_command_receiver_ignores_old_episode_and_timeout_only_requests_relaxation
.venv/bin/python3 scripts/report_arm_exploration.py \
  --roots runs/arm_exploration/stall_audited_continuation \
  --output runs/arm_exploration/audited_comparison.json
.venv/bin/python3 runs/arm_exploration/analyze_audited_continuation.py
```

Тесты: **191 passed,1 deselected,6 warnings in80,04 s**. Исключён только
ранее проверенный вне sandbox UDP-loopback тест:создание socket здесь запрещено.
Это не заявление о новом полном запуске192 passed.

Продолжение с неизменными физикой/PPO/LR/reward завершилось после114688
переходов,6825 optimizer steps,1247,1 s по`train_and_validation_stage_passed`.
Отдельный deterministic assessment подтвердил **18/18 train;0,800°**,
**27/27 validation;1,027°**,**8/8 test;0,670°**,failures0 во всех наборах.
Все policy tensors/contract/timestep best и last совпали. Runner/audit
source snapshots совпали с SHA в metadata. В журнале572 завершённых
учебных эпизода;script анализа использует исходные traces,не повторный
frozen-policy replay. Test оценивался после выбора checkpoint и остаётся
известным набором прошлых опытов,а не новым независимым holdout.
Подробные результаты и ограничения —в журнале опыта;новые weights и данные
остаются ignored в`runs/arm_exploration/`.

Headless demo отдельно запущен с новым best,`--mass 2 --tilt 120
--initial-angle 135 --target 135 --seconds 4 --output
runs/arm_exploration/stall_audited_continuation/demo_135_tilt120.jsonl`:
200 native v2 telemetry строк,4,0 s,failure false,full RMSE4,713°.
Этот full RMSE включает начальный переходный процесс;tail RMSE2,442°
получен отдельно в assessment,их не смешиваем. Viewer/GLFW не запускался.
Полный gzip-аудит прочитан до конца:114688 переходов;runner/audit/exploration
source snapshots совпали также с текущими файлами. Все свои training,
assessment,analysis,test и demo jobs завершены с exit0.

## Причинное исследование при застое — 2026-11-10

Протокол: [arm-stagnation-exploration.md](arm-stagnation-exploration.md).

```bash
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m pytest -q tests/test_arm_exploration.py \
    tests/test_arm_reward_curriculum.py tests/test_arm_workload.py
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m pytest -q
```

Целевой запуск: **8 passed in7,06 s**. Полный: **189 passed,1 failed in67,44 s**;
единственная ошибка —sandbox запрещает создание UDP socket в
`test_command_receiver_ignores_old_episode_and_timeout_only_requests_relaxation`.
Этот же тест отдельно повторён с тем же resource-prefix вне sandbox:
**1 passed in0,18 s**. Это не единый полный запуск190 passed.
Проверены detector/reset,неизменность native transitions/reward,перенос функций
и RMSprop,равенство PPO forward/evaluate_actions log probabilities при pressure1,
рост variance без сдвига mean,конечные градиенты,реальное PPO-обновление и strict
checkpoint roundtrip с сохранением optimizer state. Эти проверки не устанавливают
качество движения;парное обучение/assessment выполняются отдельно.

Парный пилот завершён:по65536 переходов. Off train13/18,val14/27;7,500°,test4/8;
7,799°;stall train14/18,val17/27;4,891°,test6/8;2,714°. Все deterministic,
failures0. Best=last в обеих ветвях;при assessment проверено равенство всех
policy tensors/contract/timestep,результаты last повторно использованы.
SGD5044/4995 (target-KL активен),время475,92/482,45 с. Контракты исходной
физики/reward не менялись. Внешний read-only `/proc` probe подтвердил CPU12–15,
nice19 всех6 собственных процессов и summed RSS2,133 GiB между оценками.

```bash
# Каждая команда с тем же nice/ionice/taskset/OMP/BLAS-prefix выше.
.venv/bin/python3 scripts/train_arm_exploration.py --mode off --root runs/arm_exploration/off
.venv/bin/python3 scripts/train_arm_exploration.py --mode stall --root runs/arm_exploration/stall
.venv/bin/python3 scripts/report_arm_exploration.py --roots runs/arm_exploration/off runs/arm_exploration/stall
```

В standalone head probe100000 samples из сохранённых screenshot-alpha/beta
при pressure1 доли>95% у TRIlong/TRIlat/TRImed43,03/2,06/17,46%;средние сохранены.
Это проверка распределения,не движение. После добавления resume/adaptive stop
в runner целевые проверки повторены:**2 passed in4,88 s**,в том числе перенос
уже156-channel модели без потери tensors/counters. AST4 новыхPython-файлов passed.
Строгий policy module после пилота не изменялся;исходный runner сохранён
в каждом root,а продолжение использует новый runner snapshot.

Завершены ещё две фазы stall:131072(23/27;2,408°,stop`budget_cap`) и32768
(25/27;1,192°,stop`validation_stage_passed`). Initial каждого совпал с last
parent. Вторая фаза использовала`--steps0 --adaptive-stop` и перенесённый
reference/counter;она остановилась по gate,а не по плато. Source module
policy/physics не менялся. Итог от общего parent:229376 новых переходов.

```bash
# Все команды с тем же resource-prefix,последовательно.
.venv/bin/python3 scripts/report_arm_exploration.py \
  --roots runs/arm_exploration/stall_continuation runs/arm_exploration/stall_until_plateau \
  --output runs/arm_exploration/continuation_comparison.json
.venv/bin/python3 runs/arm_exploration/diagnose_final.py
.venv/bin/python3 scripts/demo_arm_preacher.py \
  --controller runs/arm_exploration/stall_until_plateau/best.zip \
  --mass 2 --tilt 135 --initial-angle 125 --target 125 --seconds 4
.venv/bin/python3 scripts/demo_arm_preacher.py --seconds .2
```

Все exit0. Final train16/18;1,291°,validation25/27;1,192°,отдельная
проверка6/8;2,666°,physical failures0. Полный18/18 train-gate не пройден,
экономизация не начиналась. Best/last tensors/contract/timestep совпадают
в обеих фазах;assessment last переиспользован. Пробные14 frozen-policy
прогонов(mean / base samples / broad samples) не участвовали в отборе;
все passfalse по строгому критерию,failures0. На130° broad samples уменьшили
tail error11,29→2,38…3,69°,но settled fraction лишь0,235…0,510.
На135° broad error6,24…6,83° против7,01° mean. Это иллюстрация остаточных
режимов,не новый статистический benchmark.

Новый demo checkpoint прошёл4-секундный headless запуск125→125°,mass2,tilt135:
failurefalse,full RMSE3,011°(включает переходный процесс). Старый default controller
загружен и прошёл0,2-секундный smoke безfailure;это не проверка точности движения.
Native GLFW/view/input здесь не запускались. Demo поддерживает строгие
exploration/reward loaders и явные initial/target;старые defaults сохранены.

После добавления переноса plateau-состояния целевые тесты:**3 passed in5,88 s**;
проверка защищает accumulated reference от подмены любым малым новым best.
После прогонов runner дополнен сохранением block counter до выхода по gate
и восстановлением счётчика старых gate-summary по history:это исправление
остановки будущих продолжений,не изменение выполненных policy/rollouts.
Финальный повтор целевого файла:**3 passed in4,99 s**. AST5 новых/изменённых
Python-файлов passed;для всех4 roots подтверждены завершённый status,SHA best
checkpoint,неизменный exploration source и соответствие runner snapshot его
записанному SHA. `git diff --check` без ошибок. Все собственные training,
assessment,probe и headless sessions завершились exit0;обучения в фоне не оставлено.

## Выходная Beta-head на проблемных состояниях — 2026-11-10

```bash
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 runs/problem_poses/inspect_output_head.py
```

Exit 0. Три screenshot-состояния воспроизведены; конечные углы совпали с
manifest с допуском1e−8°. Прочитаны raw logits,alpha,beta,mean,STD и производные
среднего по logits;по100000 Beta samples/состояние с seed0. Выход
`Linear(128,12)→Softplus+0.05→Beta`,deterministic mean;скрытые функции SiLU.
У трёх разгибателей при q142,48° alpha22,023/12,229/24,274,
beta5,193/19,166/10,974:floor0,05 не достигнут. P(u>0,95) соответственно
0,0064332/4,9446e−18/5,4959e−7 при фиксированном состоянии.
Хвосты добавлены отдельным проходом по сохранённым параметрам без повторной
симуляции:96-node Gauss–Legendre, независимая сверка192 nodes для всех36
интегралов (18 мышечных распределений×2 порога),max relative difference
8,01e−13,exit0. Script затем дополнен этим же расчётом для повторяемости.
Исходные данные —ignored `runs/problem_poses/output_head.json`.
Это диагностика выходов,не история нахождения действий во время PPO.
Код policy/физики и веса не изменялись;новое обучение/pytest не запускались.

## Эвристика из идентичных проваленных состояний — 2026-11-10

```bash
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 runs/problem_poses/same_state_heuristic.py
```

Exit 0. Три learned-прогона воспроизведены до 4 с, затем команды заменены
эвристикой на 2 с без вмешательства в физическое состояние/задержки. Изменён
только горизонт truncation 4→6 с при reset; на 4 с углы, скорости, активации,
силы, плечи сил, относительные длины волокон и capacity совпали с исходными
кадрами с абсолютным допуском 1e−10. Первое попадание в ±1° от исходной цели
через 0,64/0,70/0,94 с; конечные углы 124,973/119,971/29,923°, failures 0/3.
Это времена пересечения полосы, не критерий времени установления.

В отдельной диагностике копии мышц каждой исходной позы интегрированы 300 мс
при фиксированной общей длине с неизменной первой командой сети/эвристики.
Это не native-движение руки и не прогноз движущегося сустава. SHA исходных
q/qvel/observation/полных muscle state vectors до и после проб совпал.
При 142,48° net muscle torque сети/эвристики −6,543/−9,585 Н·м; при 20,52°
+7,466/+8,645 Н·м. В последней позе эвристика запрашивает BRA 42,23% вместо
29,92% сети, при немного меньших командах бицепсу. Native endpoint-моменты
антагонистов на исходных кадрах: 0,047/0,037/0,124 Н·м по модулю.
Данные и код остаются в ignored `runs/problem_poses/`; таблицы в галерее.
Обучение и код физики/контроллеров не менялись, pytest заново не запускался.

## Силы и механические рычаги на проблемных снимках — 2026-11-10

```bash
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 runs/problem_poses/inspect_forces.py
```

Exit 0. Повторены три пары четырёхсекундных прогонов: конечные углы
совпали с manifest снимков с точностью 1e−8°, активации — с абсолютным
допуском 1e−10. На каждом шаге сохранены шесть сил сухожилий, плечи сил,
моменты, активации, bias/constraint/passive-limit и состояние контакта.
Артефакт — ignored `runs/problem_poses/muscle_forces.json`; таблицы добавлены
в галерею. При цели 125° TRIlong сети даёт 555,92 Н × 4,61 мм = 2,57 Н·м,
эвристики — 330,48 Н × 8,72 мм = 2,88 Н·м: сравнение активаций разных поз
не отражает сравнение моментов. В старом E8 при цели 30° подтверждена
поддержка пассивного ограничителя 1,224 Н·м в ошибочной позе 20,52°.
Код симуляции, policy и условия не менялись; новых обучений не проводилось.

## Скриншоты проблемных положений — 2026-11-10

```bash
nice -n 19 ionice -c 3 taskset -c 12-15 env MUJOCO_GL=egl \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 runs/problem_poses/capture.py
```

Exit0. Сохранены и визуально просмотрены7 PNG1440×1080:три пары policy/heuristic
и дополнительный кадр контактных оболочек. Снимки на4-й секунде реальных
deterministic-прогонов с2 кг. Два примера —selected final position-only,
третий —E8 (это явно подписано). Наклон/условия/камера совпадают внутри пары;
camera fit уточнён для целевой/провальной позы на private render model/data.
Первый общий-ROM render тоже завершился exit0,затем заменён более читаемым fit.

Пары q policy/heuristic:target125°,tilt135° —142,476/125,015°;
130→120°,tilt120° —142,437/120,010°;
target30°,tilt45° —20,519/29,988° (E8).
Для каждого кадра SHA integration-state MuJoCo,полных muscle states,
observations и telemetry до/после рендера совпал. Это не тест GLFW-окна/ввода.
Core physics/viewer не изменялись;новые pytest для получения снимков не нужны.
PNG headers/dimensions,завершение manifest и source SHA проверены отдельным
stdlib-скриптом. Полные условия/идентичности policy —в `runs/problem_poses/manifest.json`,
галерея —`runs/problem_poses/README.md`. Все снимки и их source остаются в ignored runs/.

## Завершение поэтапной награды и активации эвристики — 2026-11-10

После paired65536-transition пилотов position-only продолжена adaptive-фазой
из last. Initial validation побитово равна parent last; stopping initial-check
не считается учебным прогрессом. После229376 переходов/17549 SGD фаза
завершилась exit0 по`validation_plateau_two_blocks`,best/last13/27;7,97351°.
Все training/evaluation выполнялись последовательно,nice19,CPU12–15,
ionice idle,Torch/BLAS1. При освобождении evaluation-пула проверены6 собственных
процессов и summed RSS2,131 GiB между проверками;affinity/nice сохранены.

```bash
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/evaluate_arm_reward_curriculum.py \
  --roots runs/arm_reward_curriculum/native runs/arm_reward_curriculum/position_only \
    runs/arm_reward_curriculum/position_continuation --workers 2 --include-test
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 runs/arm_reward_curriculum/contact_escape_source.py
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/audit_arm_reward_rollout.py \
  --controllers runs/arm_reward_curriculum/native/best.zip \
    runs/arm_reward_curriculum/position_only/best.zip \
    runs/arm_reward_curriculum/position_continuation/best.zip \
  --output runs/arm_reward_curriculum/reward_rollout_1p5kg.json --angle 125 --tilt 135 --mass 1.5
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/probe_arm_heuristic_activation.py
```

Все exit0. Train/test native10/18 и2/8;position-only pilot10/18 и2/8;
после продолжения13/18 и4/8. Test не использовался для выбора checkpoint
или продолжительности обучения;physical failures0. Схемы/контракты checkpoint
проверены loader,outputs сохранены в ignored roots. Экономизация не запускалась.
Escape-проба:три одинаковых состояния на2 с подтверждены SHA256;из142,45°
эвристика пересекла125° через0,68 с,full extensors —через0,52 с,learned не вышла.
Моментные значения относятся к endpoint,не среднему по20-ms интервалу.
Проба эвристики прошла52/52 известных провальных условий сетей;выходы
возбуждения и активации не смешивались. Максимальный sampled-пик бицепса59,67%,
трицепса≈100%;tail-max30,30% и64,35% соответственно.
Попытка построить необязательную Matplotlib-фигуру завершилась
`ModuleNotFoundError`:библиотеки нет в venv. Зависимости не устанавливались,
результаты представлены таблицами/JSON. Новых регрессионных прогонов после
изменения только runner/диагностик не добавляли;проверки reward/serialization ниже.
Финальные AST-проверки7 изменённых/новых Python-файлов прошли;для всех3
выбранных policy подтверждены SHA checkpoint,pinned workload/reward source
и сохранённого runner snapshot. `git diff --check` без ошибок.

## Поэтапная награда руки — 2026-11-09

План: [arm-reward-curriculum.md](arm-reward-curriculum.md).

```bash
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m pytest -q tests/test_arm_reward_curriculum.py tests/test_arm_workload.py
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m pytest -q
```

Первая команда:6 passed in5,00 s. Полный запуск:187 passed,1 failed in67,88 s;
единственный failure — запрет создания UDP-сокета sandbox в
`test_command_receiver_ignores_old_episode_and_timeout_only_requests_relaxation`.
Повтор только этого теста вне sandbox:1 passed in0,19 s. Это два запуска,
не единый прогон188 passed. `git diff --check` без ошибок.
Pinned SHA `arm_workload.py` совпал с контрактом родителя E10.
Новые контракт/loader/reward не делают старые checkpoint совместимыми молча.
Training-результаты записываются отдельно после фактического завершения.

## Разбор провалов по ориентации руки — 2026-11-09

Локальной `.venv/bin/python3` выполнена агрегация сохранённых `best_validation`
и `best_test` E8/E10:144 строки на модель,12 на каждый наклон. Проверены размеры
групп и48/48 pass при0,75 kg у обеих моделей. Статические моменты восстановлены
из `runs/arm_workload_audit/audit.json` через `C+A*cos(tilt)+B*sin(tilt)`;
assert остатка<1e−12 Nm выполнен для всех использованных поз. Exit0;
сводка `runs/arm_workload/angle_analysis.json`,описание E13 в журнале.
Новых симуляций/обучения не было. Менялись только документы; pytest не запускался.

Для E14 той же `.venv/bin/python3` прочитаны tail-активации E8/E10 и временные
ряды `hold_joint/preload_probe_2kg.json`; exit0. Числа в таблице — средние
трёх сгибателей/трёх разгибателей, не силы и не проценты запаса момента.

Для E15 stdlib-скриптом `.venv/bin/python3` сверены сохранённые reward terms,
суммы tail-наград и аналитический пример position/velocity; exit0. Assert
подтвердил более высокую tail-награду heuristic и разность excitation-cost
меньше1e−5 до multiplier. Использована формула с `1-exp(-1)` из
`arm_precision.py`. Результат:`runs/arm_workload/reward_balance_analysis.json`.
Физика,награда и policy не изменялись; новых training/evaluation прогонов нет.

## Ограничение фонового обучения на ноутбуке — 2026-11-09

По просьбе владельца оставлен один основной опыт `hold_joint`. Для его дерева
из 10 процессов проверены affinity CPU 12–15 и nice 19; существующие потоки
обработаны через `/proc/PID/task`, I/O приоритет установлен `ionice -c 3`.
Использованы `os.sched_setaffinity`, `os.setpriority`, затем проверены значения
через `/proc`, `ps` и `os.sched_getaffinity`. Другие пользовательские процессы
не изменялись.

Вспомогательный `fatigue_refine` завершён SIGTERM вместе с дочерними процессами
после сохранённой validation на 16384 новых шагах; exit 143 ожидаем для этой
остановки. Основной процесс продолжает работу, физика/learning rate не менялись.
`free -m`: available RAM 2179 → 5493 MiB; swap остаётся занят, принудительной
очистки не было. Повторный `ps` подтвердил отсутствие вспомогательных Python
процессов. Новых training/test/GUI запусков для этой настройки не потребовалось.

## Разные нагрузки/ориентации и затраты — 2026-11-09

План/журнал: [arm-load-orientation-efficiency.md](arm-load-orientation-efficiency.md).

Последовательные проверки после ограничения нагрузки:

```bash
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/probe_arm_workload_exploration.py \
  --controller runs/arm_workload/hold_joint/best.zip \
  --output runs/arm_workload/hold_joint/exploration_probe.json
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/evaluate_arm_efficiency.py \
  --roots runs/arm_workload/fatigue_control runs/arm_workload/fatigue_03 --workers 2
```

Обе exit0. Основное обучение временно SIGSTOP и возобновлено через SIGCONT
в`finally` без reset/RNG restart; snapshot пробы — SHA`f347eef47e27…`,
best после393216 дополнительных transitions. Детерминированная probe-wrapper
побитово воспроизвела6 сохранённых validation-строк. Всего24 эпизода:
deterministic2/6,stochastic0/18; physical failures0. В худших двух положениях
sampled-команды не исправили~18° ошибку; в остальных выросла RMS скорости.
Шум/mean — частичное отличие, а не установленная причина всего провала.

Baseline E8 завершилась по`validation_plateau_two_blocks`:458752 дополнительных
transitions,35735 SGD,best2,240°;56/72,last2,498°;50/72,failures0.
Проба`probe_arm_workload_preload.py --angle 125 --tilt 135 --mass 2`
на этом best: learned cold/equilibrium17,907°/17,912°,
heuristic0,01389°/0,01550°. Reset-preload сохранилq/qvel/time и дал
статический баланс−9,271183 Nm приaext0,442807; learned сразу снижаетu до0,3007.
Это диагностические reset-условия, не помощь policy во время обучения.

Перед E10 повторено`python3 -m pytest -q tests/test_arm_workload.py` в проектной
venv с nice19/CPU12–15/одним BLAS потоком: **4 passed in3,74 s**.
Runner E10 с`--hold-targets 30 60 90 120 135 --hold-start-offsets 0 10`
создал9 учебных пар/324 комбинации, initial validation72 побитово равна best
родителя E8. Новая training-фаза ещё не является успешной motor policy;
actual runner source/spec/parent SHA сохранены в её root.

E10 затем завершилась exit0 по`validation_plateau_two_blocks`:131072 новых
transitions/10240 SGD,best65536 —2,098°;57/72,last2,214°;55/72,
failures0. Проверки не меняли PPO,наград,наблюдений или исходной механики.

```bash
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/probe_arm_workload_coverage.py \
  --controllers runs/arm_workload/hold_joint/best.zip \
  runs/arm_workload/hold_joint_corrective/best.zip \
  --output runs/arm_workload/coverage_probe.json --workers 2
```

Exit0.12 predefined **train**-условий:hold120/135°×tilt120/150°×0,5/1/2 kg,
capacity1,4 s. Принадлежность каждой ячейки training-spec проверена.
E8/E10:5/12;7,118° и6/12;6,843°. Native heuristic:12/12;0,01298°.
Physical failures0. Провал есть на самих учебных условиях,а не только между
углами validation. После двух плато совместное обучение остановлено;
вся последующая evaluation использует уже выбранные frozen snapshots.

```bash
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/evaluate_arm_workload.py \
  --roots runs/arm_workload/hold_joint runs/arm_workload/hold_joint_corrective \
  --kind hold_joint --workers 4
```

Exit0. Отдельный заранее определённый test72 —holds30/60/90/120°,
6 промежуточных tilt,массы0,75/1,5/2 kg,capacity1. Обе selection frozen до test;
strict replay best/last побитово равен сохранённой validation у обеих моделей.
E8 initial/best/last:5,634°;25/72 /2,650°;54/72 /2,878°;51/72.
E10 initial/best/last:2,650°;54/72 /2,518°;52/72 /2,640°;53/72.
Все432 test-эпизода без physical failure. Выбор checkpoint по test не менялся.
При более низкой средней ошибке E10 best прошла меньше удержаний,чем E8 best;
универсальное управление разными грузами/ориентациями не получено.

```bash
nice -n 19 ionice -c 3 taskset -c 12-15 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/demo_arm_preacher.py \
  --controller runs/arm_workload/fatigue_03/matched_efficiency.zip \
  --mass 2 --tilt 30 --seconds 16 --scripted \
  --output runs/arm_workload/fatigue_03/demo_2kg.jsonl
```

Exit0,failure false,full-transition RMSE20,596° с изменениями целей.
Телеметрия проверена:800 последовательных v2-кадров,один episode,
simulation time0,02→16,00 s. Это headless-проверка загрузчика/непрерывности,
а не новая оценка GUI и не ошибка установившегося удержания.
`scripts/report_arm_workload.py --plot` с доступным локальным Matplotlib
завершился exit0; сводка/PNG/SVG обновлены после финальных evaluation.
Фигура из собственных агрегированных метрик сохранена как
`docs/images/arm-workload-validation.svg`; геометрия/веса/rollouts в неё не включены.
`py_compile` пяти изменённых training/evaluation/probe scripts и`git diff --check`
завершились exit0. Физика,шаги,добавленные155 входов и checkpoint-source hash
`arm_workload.py` за это продолжение не менялись; новых зависимостей не установлено.

Efficiency: выбор обеих моделей frozen до test, одинаковые selector/parent/
PPO/budget/validation; strict replay совпал у обеих. Test control2,008°;25/33,
cost1,831°;26/33; mean active fraction−4,26%,mean a²−19,30%.
Четыре20-s holds обе4/4:0,636/0,709°,active fraction−1,06%.
Maximum per-muscle capacity loss0,062207→0,047442,но mean net-loss всех мышц
снизилась только на1,25%,а coactivation выросла.10%-gate active fraction
не достигнут. Полные числа/ограничения сравнения — в журнале E7.

Исходное состояние сохранено коммитом `8ea11f9`. Старые physics/observations/
policy contracts не менялись; новая среда/155-channel checkpoint отдельные.

```bash
.venv/bin/python3 -m pytest -q tests/test_arm_workload.py
taskset -c 12 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m pytest -q
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m pytest -q \
  tests/test_telemetry.py::test_command_receiver_ignores_old_episode_and_timeout_only_requests_relaxation
taskset -c 6 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/audit_arm_workload.py
```

Targeted: **4 passed in2,76 s**. Проверены точные native transitions при30/210°,
1/2 kg, периодичность+360°, сохранение dynamics при новой cost и перенос
старой функции/RMSprop в155 inputs с нулевыми новыми колонками, strict reload.
Full suite: **185 passed,1 failed,6 warnings in190,44 s** при параллельном
аудите/пробных процессах. Единственный failed — sandbox запрещает UDP socket.
Повтор вне sandbox: **1 passed in1,14 s**. Все186 проверок выполнены;
это не один полный186-pass запуск. Пакеты не менялись.

Аудит exit0: масса подвижной руки без груза1,783386 kg; humerus1,99757 kg
закреплён. Из672 статических условий2 недостижимы для разгибателей:
135°,2,5 kg,capacity0,7,tilt120/150°. На изученных≤125° и≤2 kg статический
запас есть. Исходная mean tail RMSE шести20-s удержаний1,878°,
zero-extensor intervention18,688°. Это отключение команд без переобучения
сгибателей, не доказательство обязательной коактивации.

Smoke:

```bash
taskset -c 0,2,4,6 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/train_arm_workload.py --stage mass \
  --root runs/arm_workload_smoke --fixed-steps 1024 --check-steps 1024 --minimum-steps 32768
```

Exit0,4 среды×256,1024 transitions,80 SGD. Это plumbing, не успешный контроллер.
Initial validation27 условий7,819692°;10/27, after7,901573°;10/27, failures0.
Первые66k-опыты используют параллельную evaluation; smoke был с serial evaluation.
Гиперпараметры, hashes parent/runner, версии, reset/sampling schedule и spec
сохраняются в metadata. Actual runner source архивируется в каталоге опыта.
Веса/rollouts в ignored `runs/arm_workload*`.

Выполнены:

```bash
taskset -c 0,2,4,6 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/train_arm_workload.py --stage mass \
  --root runs/arm_workload/mass --minimum-steps 65536
taskset -c 8,10,12,14 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/train_arm_workload.py --stage fixed \
  --root runs/arm_workload/fixed_control --fixed-steps 65536
```

Обе initial validation равны smoke:7,819692°;10/27. Mass завершён после98304
по success gate:1,963°;25/27. Fixed-control65536:best6,891°;10/27.
На одинаковых65536 mass1,965°;22/27 против fixed last7,094°;9/27.
После freeze strict replay best/last обеих веток совпал побитово;
test33 при0,75/1,5/2,5 kg:mass2,658°;26/33, fixed best9,631°;8/33.
Четыре20-s удержания приcapacity0,85 не пройдены (mass6,624°;0/4).
Команда final evaluation:

```bash
taskset -c 8,10,12,14 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/evaluate_arm_workload.py \
  --roots runs/arm_workload/fixed_control runs/arm_workload/mass --kind mass --long
```

Отдельная ветка `mass_capacity` обучена сcapacity0,7/1,0, прежними массами/
заданиями/reward. После163840 новых transitions validation при0,85:
**0,973°;25/27;failures0**,12800 SGD, `validation_stage_passed`.
Ближние ориентации `hold_near`:147456 новых transitions,1,187°;11/12,
failures0; success gate. Полный круг продолжается. Первоначальный процесс
`hold_circle` завершился внешним SIGTERM/143; причина неизвестна. Последний
checkpoint131072 новых transitions строго загружен в `hold_circle_resume`;
weights/moments перенесены, среды/RNG перезапущены явно.

Диагностика худшего125°/tilt135° на промежуточном circle checkpoint65536:
`scripts/probe_arm_workload_preload.py`, exit0. Cold/preloaded learned дают
17,68667°/17,68670°, heuristic0,00899°/0,00687°. Проверены неизменностьq/qvel/
time при reset-preload и статический баланс−6,007326 Nm. Это диагностическое
начальное состояние, не изменение runtime-динамики или помощь обучению.

Offscreen `MUJOCO_GL=egl`: фактически отрисованы и просмотрены PNG при
tilt30/120/210/300°,75° elbow,1 kg после1 s heuristic. Камера охватывает
кости/груз/мышцы и сохраняет отдельные цветовые шкалы; изображения в ignored
`runs/arm_workload/render_review/`. Это проверка визуализации ориентаций,
не проверка новой PPO-policy или клавиатурного GUI.
Headless CLI `scripts/demo_arm_preacher.py --controller runs/arm_workload/mass/best.zip --mass 2 --tilt 30 --seconds 16 --scripted`:
exit0,16 s без failures/reset, telemetry800 кадров. Full-transition RMSE21,548°
включает четыре смены цели и не заменяет tail-метрики.
Текущая сводка:
`.venv/bin/python3 scripts/report_arm_workload.py`.

Продолжение E4 final evaluation выполнено с`--roots runs/arm_workload/mass_capacity
--kind mass --long --workers 1`: exit0,strict validation replay best/last
совпал с training evaluation четырёх workers. Общий fresh mass-test:
1,782°;26/33;failures0. Четыре20-s удержания приcapacity0,85:
**0,860°;4/4**,full-episode RMSE0,343/0,638/1,921/1,205°,
установление0,56/0,66/0,40/0,30 s. Максимальная фактическая потеря capacity
0,05673 у BICshort; physiological calibration этим не подтверждена.

Тот же scripted CLI демо с`--controller runs/arm_workload/mass_capacity/best.zip`
и`--mass 2 --tilt 30 --seconds 16 --scripted` завершился exit0 без failure.
Телеметрия`mass_capacity/demo_2kg.jsonl`:800 кадров, единственный episode0,
время0,02→16,00 s, последнее положение46,24° при цели45°.
Full-transition RMSE20,495° включает смены цели и не является ошибкой удержания.
Нового GUI smoke для этой policy не запускали.

Полный круг E5 вышел на плато: первая фаза131072 сохранённых transitions,
продолжение196608;best2,062°;20/24. Уточнение только LR3e−5→1e−5 из этого
best,32768 transitions/2560 SGD,241,35 s:1,866°;22/24;failures0.
Прошёл practical validation-gate, но125°/tilt135° остаётся с17,356° ошибки.
Не смешивать этот критерий с успешностью каждого угла.

E6: обе парные ветки2×512 завершены после65536. Initial policy/optimizer/
pytorch tensor archives совпали побайтово. Control4987 SGD,1526,86 s;
activation²-cost5066 SGD,1505,67 s. Efficiency-best со штрафом остался initial;
снижение затрат без потери accuracy/pass не установлено. Tracking-best
1,673°;25/27,mean a²0,030824 (initial0,030475);last2,315°;22/27.
Код runner при E6 архивирован до более поздней смены efficiency-selection.

```bash
taskset -c 1,3,5,7 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/audit_arm_workload.py --effort-reference
```

Exit0: native heuristic на validation27 даёт0,232°;27/27,a²0,009137,
active fraction0,060359; на20-s holds0,00479°;4/4,a²0,010383,
active fraction0,057652. Это reference, не доказанный optimum.

Postprocessing compile/diff checks прошли. Для evaluator проверено совпадение
полного uncompressed ZIP+sidecar содержимого best/last E4 иinitial/energy-best E6
(исключены только ZIP timestamps и внешняя SHA). Новая версия переиспользует
evaluation только для таких идентичных архивов и одинакового списка условий;
файловая SHA/metadata и strict replay результата по-прежнему проверяются.
Reporter больше не импортирует training/Torch ради чтения JSON. Кривые
группируются по точному validation-набору; по осиX — lifetime transitions,
чтобы последовательные curriculum-фазы не выглядели запуском с одного parent.

## Интерактивное PPO-демо руки — 2026-11-09

[Запуск и клавиши](arm-interactive-demo.md). Launcher `scripts/demo_arm.sh`
выбирает tested Python и прежний лучший none-controller
`runs/arm_continuation/best.zip`; физика/observations/reward/веса не менялись.
В этом позиционном viewer trigger фиксирован1, release/trigger keys выключены;
generic viewer в остальных режимах сохраняет их. Ручные цели не перезаписываются
автоматическим сценарием, если явно не указан `--scripted`.

```bash
.venv/bin/python3 -m pytest -q tests/test_demo.py tests/test_vertical_viewer.py
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m pytest -q
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m pytest -q \
  tests/test_telemetry.py::test_command_receiver_ignores_old_episode_and_timeout_only_requests_relaxation
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash scripts/demo_arm.sh --seconds 3
.venv/bin/python3 -m py_compile src/myo/demo.py scripts/demo_arm_preacher.py
bash -n scripts/demo_arm.sh
git diff --check
```

Targeted: **18 passed in0,86 s**. Старый keyboard-сценарий расширен fixed-trigger
вариантом; регрессия ожидания native cleanup вертикального viewer дополнена
рукой. Проверяются сохранение реальной физики при GUI-edits и блокирование
`0` только в fixed-trigger режиме. Новых dynamics/schema тестов не добавлялось.
Full suite: **181 passed,1 failed,6 warnings in63,70 s**. Единственный failed —
создание UDP-сокета, запрещённое sandbox. Повтор этого теста вне sandbox:
**1 passed in0,14 s**. Все182 проверки выполнены, это не один182-pass запуск.
Compile, shell syntax и diff-check: exit0.

На реальном X11 desktop `:1` проверены окно, отрисовка и программное закрытие
через3 simulated seconds. Из sandbox X11 недоступен; GUI smoke выполнялся
вне sandbox. Первый запуск выявил зависание после public passive-handle close;
его тестовый процесс завершён и заменён исправленным запуском. Исправление:
как в `VerticalViewer`, собственный render-thread вокруг pinned MuJoCo3.17
`_launch_internal`, ожидание `join()` до освобождения model/data и завершения
GLFW. **Повторный GUI smoke exit0**, без зависания и physical failure.

Отдельный inline прогон через настоящий `DemoViewer.poll()` с очередью
клавиш↑/↓ и фиксированным trigger выполнил75→20→105→45° по200 шагов.
Команды`0/-/=` не изменили trigger1. Во всех800 шагах запрошенная цель
оставалась ручной, **16 s без reset/failure**, с родной динамикой задержки
и series elasticity. Tail RMSE: **0,527/1,496/2,745/0,101°**.
Результат — ignored `runs/interactive_arm_demo/controls.json`.

Через `MUJOCO_GL=egl`, существующие временные PIL dependencies
(`PYTHONPATH=/tmp/myo-supervised-plot-deps`) и `myo.demo.render_frame`
сняты реальные конечные позы каждого участка,1280×960:
`runs/interactive_arm_demo/goal_{75,20,105,45}.png`. Кадры75° и105° просмотрены:
кости, отдельные объёмы брюшек/сухожилий, нагрузка1 kg, числовые сигналы,
цветовые шкалы и правильная fixed-trigger подсказка видны, текст не перекрывается.
Это процедурное изображение, не UE/Flesh/skin. Физические клавиатурные события
на X11 не инжектировались: управление проверено на callback-очереди и real env.

## PPO с absent/old20/new60 признаками — 2026-11-09

Протокол: [arm-anticipation-ppo.md](arm-anticipation-ppo.md).
Новый policy/extractor и strict loader вынесены в отдельный модуль;
source-pinned `arm_precision.py`, `arm_learning.py`, `arm_anticipation.py`
и физика не менялись. Демо научилось строго загружать новый тип policy.

```bash
.venv/bin/python3 -m pytest -q tests/test_arm_feature_comparison.py
taskset -c 6 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m pytest -q
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m pytest -q \
  tests/test_telemetry.py::test_command_receiver_ignores_old_episode_and_timeout_only_requests_relaxation
.venv/bin/python3 -m py_compile src/myo/arm_feature_comparison.py \
  scripts/compare_arm_anticipation_ppo.py scripts/report_arm_anticipation_ppo.py \
  scripts/demo_arm_preacher.py
git diff --check
```

Новый regression: **3 passed in3,38 s**, проверены сохранение mean/value и
RMSprop moments при подключении предиктора, неизменность frozen-весов при
изменении private-весов после PPO, roundtrip и запрет подмены frozen encoder.
Полный suite: **179 passed,1 failed,6 warnings in60,02 s**.
Единственный failed — создание UDP-сокета в sandbox: `PermissionError`.
Этот тест повторён отдельно вне sandbox: **1 passed in0,23 s**.
Таким образом, выполнены все180 проверок; это не один полный180-pass запуск.
Compile/diff — exit0.

Завершены три headless CPU-ветки на ядрах0/2/4 с одним Torch/BLAS потоком,
общим родителем212992, LR3e−5, по65536 новых переходов.
Все три начальные validation совпали точно: **1,3147004645660703°;9/9**.
Test не использовался для выбора; все initial/best/last hashes совместно
зафиксированы в `frozen_selection.json` до открытия test.

Фактические train-команды (exit0):

```bash
taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/compare_arm_anticipation_ppo.py train --kind none
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/compare_arm_anticipation_ppo.py train --kind old20
taskset -c 4 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/compare_arm_anticipation_ppo.py train --kind new60
taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/compare_arm_anticipation_ppo.py evaluate
```

Всего196608 новых переходов; `fixed_equal_budget`, не плато.
Trainable parameters172941 во всех ветках; frozen parameters55700/55700/73748.
SGD5000/4996/5043, у каждой2/64 KL-stop; wall с validation792,98/809,80/811,83 s.
Best validation: none1,148°;9/9, old20 1,196°;9/9, new60 initial1,315°;9/9.
Last validation:1,148°;9/9,1,202°;8/9,1,632°;8/9.
Строгие загрузки и точный replay best/last validation прошли, frozen digest
неизменен. Вся none validation-история и SGD точно воспроизвели прежний
none_3e5 запуск. Общие начальные trainable weights и полные RMSprop states
трёх веток дополнительно сравнены по именам параметров: равны.

Best test: none**2,264°;8/11**, old20**2,125°;9/11**, new60**1,940°;10/11**,
но последний — initial с нулевым readout предиктора, а не выученная польза.
Last test: **2,264°;8/11 /2,144°;8/11 /2,709°;7/11**.
Все без physical failures. New60-last не прошёл удержания120/130/15°
и подъём60→125°, то есть4/5 движений; none/old20-best проходят5/5.
Прежний демо-controller не заменён, GUI не открывался.

Артефакты/веса/rollouts в ignored `runs/arm_anticipation_ppo/`.

После выбора выполнены (exit0):

```bash
taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/report_arm_anticipation_ppo.py --plot
taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/demo_arm_preacher.py \
  --controller runs/arm_anticipation_ppo/new60/last.zip --seconds 16 --scripted \
  --output runs/arm_anticipation_ppo/new60/demo.jsonl
```

Прежние временные matplotlib dependencies; пакеты проекта не менялись.
График просмотрен; после уточнения легенды new60-best как initial/zero readout
функция `plot` отдельно выполнена повторно, без повторного расчёта диагностики.
На5400 train-only наблюдениях new60-last h→mean меняет excitation в среднем
на0,00050; validation1,632°;8/9→1,791°;8/9, h→zero1,856°;7/9.
Old20-best mean-ablation1,196→1,286° при сохранённых9/9.
Замороженные признаки используются с небольшим эффектом, но это не означает
превосходства над исходным parent или none-контролем.

Все три selected-best исполнили16 s с75→20→105→45° без reset/failure.
Настоящий CLI new60-last также исполнил16 s без failure;800 кадров v2,
один run/episode, sequence0…799, шаг20 ms, physics1 ms и время непрерывны.
Tail ошибки0,601/1,081/3,391/0,948°; full RMSE20,900° включает переходы.
Финальный inline-аудит `evaluation_audit.json`: общие initial trainable
weights/full optimizer states, все selection hashes,65536 переходов на ветку,
побитовое совпадение **всей** none validation/SGD истории с прошлым запуском,
непрерывность demo — passed. GUI не проверялся.

## Увеличенный бюджет прямого предиктора60 ms — 2026-11-09

Протокол до запуска: [arm-anticipation60-continuation.md](arm-anticipation60-continuation.md).
Core-модули физики, наблюдений, reward и checkpoint schema не менялись.
Новый runner использует прежний save/load с provenance продолжения.

```bash
.venv/bin/python3 -m pytest -q tests/test_arm_anticipation_resume.py tests/test_arm_anticipation.py
.venv/bin/python3 -m py_compile scripts/continue_arm_anticipation.py scripts/compare_anticipation_budgets.py
git diff --check
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/continue_arm_anticipation.py
```

Тесты: **5 passed in2,04 s**, включая конкретную регрессию resume:
исходный и восстановленный Torch RNG, minibatch draws, полные RMSprop moments
и веса после следующего update совпали точно. Compile/diff: exit0.
Полный suite заново не запускался: формат/core не менялись, предыдущий полный
прогон176 проверок описан ниже.

Запущено продолжение150000→максимум450000, LR1e−4. Данные/normalization/sampling
те же; референс progress восстановлен из прежней validation-истории.
Завершено с exit0: **300000 дополнительных updates**, общий номер450000,
best**448000**. Продолжение с validation574,43 s CPU.
Остановка `maximum_updates`, не плато: последнее значимое улучшение448000.
Train balanced MSE на фиксированных8192 окнах0,023055→0,005690,
full validation balanced MSE0,027753→0,009601, уменьшение65,4%.

После фиксации hash и строгого replay validation выполнены:

```bash
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/report_arm_anticipation.py \
  --root runs/arm_anticipation60_extended --diagnostics --plot
.venv/bin/python3 scripts/compare_anticipation_budgets.py
```

Обе команды exit0, график просмотрен. Использованы прежние временные
matplotlib dependencies, пакеты проекта не менялись.
Test на тех же28710 окнах: угол**0,1559→0,1009°**, скорость**3,441→2,229°/s**;
старый×3 по-прежнему0,1800°/3,999°/s. Повторный direct150k точно воспроизвёл
все старые test-метрики, включая отдельные источники и20 каналов.

В22 новых deterministic policy-эпизодах на неизменённых11 TEST_CASES и
двух прежних controllers: none q/v0,2080°/5,597°/s→**0,0910°/3,081°/s**,
frozen0,1827°/5,708°/s→**0,0699°/2,752°/s**. Identities controllers и
persistence-метрики совпали точно с исходным опытом.
В16 парах команд ошибка контраста q/v0,1137°/9,072°/s→0,0666°/6,353°/s;
старый×3 всё ещё точнее по эффекту скорости3,367°/s.
Нативные исходы/33 входа/команды32 ветвей совпали с исходными записями побитово.

Финальный inline-аудит сохранён в `evaluation_audit.json`: initial parameters
и полный RMSprop state/paramgroups совпали с родителем; mean/STD совпали
с train-only пересчётом; selected hash, минимум/replay validation и
parent/dataset hashes проверены. Optimizer step counters совпали:
initial150000, best448000, last450000. Нормировки, данные и LR не менялись.
Проверенные версии: torch2.15.0+cpu, stable-baselines3 2.9.2,
MuJoCo3.17.0, NumPy2.5.4.

Артефакты ignored: `runs/arm_anticipation60_extended/`.
Новые test/probe/counterfactual labels не использовались в train или выборе best.
PPO с новым encoder не обучался, demo controllers не изменены.

## Прямой предиктор ощущений60 ms — 2026-11-09

Протокол: [arm-anticipation60.md](arm-anticipation60.md).
Новые standalone predictor/trainer/report; core физики и PPO не менялись.

```bash
.venv/bin/python3 -m pytest -q tests/test_arm_anticipation.py
taskset -c 4 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m pytest -q
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m pytest -q tests/test_telemetry.py::test_command_receiver_ignores_old_episode_and_timeout_only_requests_relaxation
.venv/bin/python3 -m py_compile src/myo/arm_anticipation.py src/myo/train_arm_anticipation.py scripts/report_arm_anticipation.py
git diff --check
```

Новые тесты: **4 passed in2,04 s**. Полный pytest в песочнице:
**175 passed,1 failed,6 warnings in45,64 s**. Единственный failure —
`PermissionError` создания localhost UDP-сокета в старом telemetry-тесте.
Повтор именно этого теста вне сетевой песочницы: **1 passed in0,20 s**.
Таким образом выполнены все176 проверок; полный suite целиком вне песочницы
не перезапускался. Compile/diff: exit0.

Обучение запущено локально, без новых зависимостей:

```bash
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 -m myo.train_arm_anticipation
```

Данные124200 переходов/621 эпизод дали66726 train-окон,27522 validation-окон.
Label60 —фактический третий следующий кадр, inputs33 только из текущего
состояния и двух FIFO-команд; новая команда подаётся в голову после encoder.
Проверены непрерывность физических каналов/FIFO и3 valid-перехода каждого окна.
Нормировки получены только из train; старые dataset/checkpoints не меняются.
Завершено с exit0: **150000 updates**, best150000, **335,22 s** CPU с validation.
Train/validation balanced MSE на последнем шаге0,02306/0,02775;
validation снизилась с1,55547. Остановка `maximum_updates`, не плато.
Выбранный hash сохранён до test, строгий reload и replay validation совпали.
Test28710 окон: direct q60/v60 **0,1559°/3,441°/s**, прежний one-step×3
**0,1800°/3,999°/s**, persistence2,1650°/20,994°/s.
Активации хуже прежнего ×3: RMSE0,00855–0,00956 против0,00132–0,00417.
Поканальные значения и отношение ошибки к изменениям20/60 ms сохранены в отчёте.

После фиксации best:

```bash
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/report_arm_anticipation.py --diagnostics --plot
```

Exit0, график просмотрен. Использованы прежние временные matplotlib dependencies;
пакеты проекта не менялись. В16 native-состояниях из8 test hold/ramp-эпизодов
выполнены32 ветви с разными текущими командами. Переигранный префикс дал
побитовое совпадение33 входов; input не конструировался из future labels.
Ошибка контраста положения0,1137° против0,2051° у старого ×3; скорости
9,072 против3,367°/s. RMS реального эффекта0,1852°/26,028°/s.
Знак скорости верен во всех16 пробах у обеих моделей; это не проверка всего Jacobian.

Дополнительно собраны22 новых deterministic policy-эпизода без обучения:
11 TEST_CASES у none212992 и frozen_3e5 196608, по2178 forecast-окон.
На none direct q/v0,2080°/5,597°/s против ×3 0,1248°/3,127°/s;
на frozen —0,1827°/5,708°/s против0,0906°/1,947°/s.
Польза нового представления для текущего управления не подтверждена.
Новые probe/counterfactual labels в train не переносились; PPO не запускался.
Финальный inline-аудит (`evaluation_audit.json`) подтвердил побитовое совпадение
mean/STD с пересчётом только train delta60, неизменность выбранного hash и
исходного dataset, минимум validation у best и одинаковые state_dict best/last.

Версии: Python3.12, torch2.15.0+cpu, stable-baselines3 2.9.2, MuJoCo3.17.0,
NumPy2.5.4. Все веса/labels/графики ignored в `runs/arm_anticipation60/`;
прежние checkpoints, core dynamics/observations/reward и demo не менялись.

## Frozen-предиктор и меньший LR — 2026-11-08

Протокол до запуска: [arm-predictor-lower-lr.md](arm-predictor-lower-lr.md).
Четыре продолжения по65536 новых переходов: none/frozen и LR1e−4/3e−5.
Внутри пары один родитель, между парами родители разного уровня обучения.
Core-модули физики/наблюдений/награды/сериализации не менялись.

```bash
.venv/bin/python3 -m pytest -q tests/test_arm_predictor_lr.py tests/test_arm_continuation.py tests/test_arm_conditioning.py
.venv/bin/python3 -m py_compile scripts/compare_arm_predictor_lr.py scripts/report_arm_predictor_lr.py
git diff --check
```

**5 passed in4,24 s**. Новый параметризованный тест проверяет сохранение weights
и RMSprop moments при смене LR, schedule/paramgroups, строгую загрузку обоих
форматов и сохранение нового LR при следующем update. В первом запуске два
новых случая выявили отсутствие `set_env` в самом тесте после reload;
после подключения возвращённой среды повторный запуск прошёл полностью.
Compile и diff: exit0. Полный pytest не запускался.

Четыре train-команды из протокола запущены локально на CPU2/4/6/8 с одним
вычислительным потоком каждая. Frozen_3e5 перенесена с медленного CPU8 на CPU0
через `taskset -apc 0 741356` без перезапуска процесса. Бюджет и выбор best не зависят от test.
Все четыре процесса завершились с exit0 и `fixed_equal_budget`: по65536
переходов, всего262144. Стартовые weights/buffers/RMSprop moments внутри
каждой пары побитово одинаковы, кроме LR в paramgroups; physics/predictor
identities совпадают. Это проверено отдельным inline-аудитом initial archives.
На каждом validation-check predictor/head/normalization/gain оставались
побитово неизменными; исходные архивы не менялись.
Проверенные версии: torch2.15.0+cpu, stable-baselines3 2.9.2,
MuJoCo3.17.0, NumPy2.5.4; обучение CPU, по одному потоку.

После завершения обучения выполнены:

```bash
taskset -c 6 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/compare_arm_predictor_lr.py evaluate
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/report_arm_predictor_lr.py --plot --diagnostics
```

Обе команды exit0. До test сохранены hashes initial/best/last всех четырёх веток.
Строгие loaders проверили conditioning identity и LR; replay best/last validation
совпал с записанным, rank best совпал с минимумом истории. Тестирование не меняло
выбор моделей. График просмотрен; использованы прежние временные matplotlib
dependencies, пакеты проекта не менялись.

| Ветка | Best validation, °;pass | Best test, °;pass | Last test, °;pass | Новых SGD |
| --- | --- | --- | --- | ---: |
| none_1e4 | 1,315;9/9 | 1,940;10/11 | 3,603;4/11 | 752 |
| none_3e5 | 1,148;9/9 | 2,264;8/11 | 2,264;8/11 | 5000 |
| frozen_1e4 | 4,351;6/9 | 4,483;5/11 | 4,118;3/11 | 3862 |
| frozen_3e5 | 1,635;7/9 | 3,492;5/11 | 3,492;5/11 | 5120 |

Физических failures0. Ниже80 SGD на rollout получили63/64,2/64,25/64,0/64
циклов соответственно. Все пять test-перемещений прошли у обеих none,
три из пяти —у обеих frozen. У none_3e5 потеряны удержания30/130°;
набор провалов frozen общий:15/30/120/130° и движения105→25°,60→125°.
Сравнение относится к одному seed и знакомому test; разные родители не
изолируют причинную пользу предиктора. Frozen_3e5 улучшался на последней
проверке, поэтому бюджетная остановка не объявляется плато.

Для всех четырёх best выполнена следующая команда с соответствующим trial
(CPU0/2/4/6, каждый с OMP/OPENBLAS/MKL=1):

```bash
.venv/bin/python3 scripts/demo_arm_preacher.py \
  --controller runs/arm_predictor_lr/frozen_3e5/best.zip --seconds 16 \
  --output runs/arm_predictor_lr/frozen_3e5/demo_best.jsonl
```

Все четыре exit0, failure false. По800 v2-кадров, один episode, непрерывные
sequence/время20 ms без reset; команды75/20/105/45° по4 s.
Tail RMSE этих четырёх участков у frozen_3e5:0,219/0,309/2,102/2,019°;
у none_3e5:0,081/1,137/2,133/1,482°. GUI не запускался.

Feature-аудит использовал5400 общих исторических train-состояний. У frozen_3e5
h1→mean меняет возбуждения в среднем на0,02613, validation1,635→1,870°,
pass остаётся7/9; h1→zero —1,896°;7/9. Случайная перестановка векторов меняет
возбуждения на0,03737, exact joint KL2,440. Это влияние, но не доказательство
сильной полезности предобучения. Веса и buffers после вмешательств не менялись.
Проверены также два stochastic streams на четырёх фиксированных заданиях:
у frozen_3e5 mean20→75°0,219° против samples1,849/3,388°; зависимости успешного
подъёма исключительно от sampling на этих пробах нет.

Артефакты остаются ignored в `runs/arm_predictor_lr/`. Прежний demo checkpoint
не заменён, код физики/награды и serialized contracts не менялись.

## Продолжение руки с LR0,0001 — 2026-11-08

Протокол и результаты: [arm-low-lr-continuation.md](arm-low-lr-continuation.md).
Core-модули физики, наблюдений, награды и сериализации не менялись.
Полный pytest в этом продолжении не запускался; выполнены два новых behavioral
теста и проверки реального обучения, загрузки и исполнения.

```bash
.venv/bin/python3 -m pytest -q tests/test_arm_continuation.py
.venv/bin/python3 -m py_compile scripts/continue_arm_precision.py scripts/report_arm_continuation.py
git diff --check
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python3 scripts/continue_arm_precision.py
taskset -c 6 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/report_arm_continuation.py --measure-probes --plot
.venv/bin/python3 scripts/demo_arm_preacher.py \
  --controller runs/arm_continuation/best.zip --seconds 16 \
  --output runs/arm_continuation/demo_best.jsonl
```

Тесты: **2 passed in2,95 s**. Они проверяют сохранение RNG/физического
эпизода/весов/счётчика optimizer при stochastic-validation и отсутствие ложного
прогресса от специализации на одной цели. Compile и diff: exit0.
Для рисунка использованы ранее установленные временные matplotlib dependencies;
зависимости проекта не менялись. Рисунок просмотрен.

Обучение завершено с exit0, `validation_all_cases_passed`: родитель163840 →
best212992, **49152 новых перехода,1410 SGD**,432,75 s с промежуточными проверками.
Validation **4,233°;3/9 →1,315°;9/9**, equal-goal RMSE4,489→1,552°.
Test после фиксации выбора: **4,002°;4/11 →1,940°;10/11**, failures0.
Все пять test-перемещений прошли; единственный провал —15→15°,5,668°.
Ни один прежний pass не потерян. Это успешная остановка для просмотра,
не свидетельство плато или полного D1.

Отдельной парной проверкой из одного checkpoint установлено точное совпадение
одного `learn(2048)` и двух `learn(1024)` при `reset_num_timesteps=False`:
все tensors policy, RMSprop state/paramgroups, Torch/NumPy RNG и native observation
совпали, счётчик8131. Результат — `runs/arm_continuation/loop_equivalence.json`.
Начальные tensors и полный optimizer state совпали с родителем; исходный архив
не изменён. Строгие reload/replay прошли; best/balanced_best/last имеют одинаковые
обученные параметры. Пробы использовали изолированный RNG и не выбирали модель.
Четыре задания проб пересекаются с историческим test; независимым новым набором
он не является.

Непрерывное headless-демо16 s прошло без failure и reset:800 v2-кадров,20 ms,
цели75/20/105/45° по4 s. Последняя секунда каждого участка имеет RMSE
0,527/1,496/2,744/0,100°; полная RMSE с переходами21,60°.
GUI не запускался. Артефакты остаются ignored в `runs/arm_continuation/`.

## Проверка гипотез о плохом обучении руки — 2026-11-08

Протокол и поэтапные выводы: [arm-hypotheses.md](arm-hypotheses.md).
Core-модули динамики/observations/reward/serialization не менялись. Новый полный
pytest не запускался: добавлены диагностические scripts и runner существующего
PPO; проверены реальные переходы, updates и строгие reload ниже.

```bash
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 scripts/check_arm_hypotheses.py gradient
taskset -c 4 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 scripts/check_arm_hypotheses.py concentration
taskset -c 6 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 scripts/check_arm_hypotheses.py critic
taskset -c 4 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 scripts/check_arm_exploration_duration.py
taskset -c 4 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 scripts/check_arm_exploration_duration.py --concentration-multiplier .2 --output runs/arm_hypotheses/exploration_wide.json
```

Все exit0. Gradient31,96 s:5 свежих1024-step rollout/update,80/12/4/17/1
реальных SGD для initial none/none best/gain0,25/raw frozen/vertical best.
Захвачены pre-clip norms actor/critic, необъяснённых обучаемых gradients нет,
веса копий действительно менялись, исходные checkpoints не перезаписывались.
В проверенных updates critic не доминирует>90% нормы; у gain0,25 median share10,3%.

Concentration55,92 s:36 полных4-s эпизодов двух arm policy,3 задания,
k4/16/64,2 streams. Условное среднее Beta сохранено на каждом шаге,
model tensors неизменны. На frozen20→75° сужениеk16 вернуло ошибку≈51,7°;
на опускании и другом подъёме точность, напротив, выросла.

Critic137,44 s:32 независимых12-s stochastic-продолжения4 состояний,
без failures. MC labels без bootstrap; V в начале20→75° ниже12-s MC
примерно на0,49/0,55. Положительный остаток return после12 s≤0,0241.
Дополнительно original native env и deepcopy после stochastic-prefix проверены
на12 одинаковых последовательных действиях: observations/rewards/dones,
q/qvel, активации, delayed excitation, fiber lengths/capacity совпали точно.
Это независимый value-audit, а не доказательство причинного ущерба от critic.

Исследование:12 native+12 wider эпизодов initial-policy, задержки/упругость
сохранены. При исходном шуме20/100/200-ms holds ни одна из4 последовательностей
не подняла руку выше старта. k0,2 с100/200-ms holds иногда дал21–24°, но ни одного
достижения75°. При20-ms draw исходные trajectories совпали с прежним аудитом.

```bash
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 scripts/resume_arm_budget_check.py
taskset -c 4 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 scripts/check_arm_update_drift.py
```

Оба exit0. Неизменённое продолжение:32768 дополнительных transitions,
153 SGD,232,85 s обучения/validation. Родительские model tensors, полный RMSprop
state, PPO scalars и contract совпали с continuation initial побитово; RNG/среда
перезапущены seed0, это не bitwise resume исторического процесса. Best139264
выбран до test:validation6,448°;4/9, test8,618°;3/11. Подъём20→75° достиг
RMSE0,166°, но ряд верхних целей ухудшился; исходный test7,334°;2/11.
Last test6,775°;2/11 не выбран заново.32/32 новых rollout с KL early stop.
Best/last/initial строго загружены, initial private/optimizer count сохранены,
native best-validation воспроизведена, parent SHA не изменился.

Update drift:3 отдельных настоящих1024-step updates, исходные weights не
перезаписаны. Whole-rollout exact KL после принятого первого шага none139264
**0,08633**, при guard0,045; это допускается штатной SB3-проверкой без rollback.
Для none131072 после12 принятых steps0,06822, gain0,25 после4 steps0,05815.
Сценарии rollout содержали только цели15/35°, query-grid включал все27 train-пар.
Snapshot none139264 с проверенным SHA сохранён только в ignored runs.

```bash
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 scripts/resume_arm_budget_check.py --root runs/arm_hypotheses/lr_1e4_none --learning-rate .0001
taskset -c 4 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 scripts/resume_arm_budget_check.py --root runs/arm_hypotheses/order_none --case-order round_robin
taskset -c 6 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 scripts/check_arm_update_drift.py --model none_low_lr --output runs/arm_hypotheses/update_drift_lr.json
```

Все exit0. Раздельные контроли LR/порядка заданий завершены из одного none131072:
оба32768 новых transitions,1997/681 SGD,328,91/320,76 s обучения с validation.
Сохранены initial tensors и полный RMSprop moments; scalars/groups отличаются
только заданным LR, у order — только последовательность27 тех же пар.
KL-stop11/32 у LR против29/32 у order и32/32 у неизменённого продолжения.
LR best163840:validation4,233°;3/9,test4,002°;4/11. Order best139264:
validation5,940°;2/9,test5,122°;3/11;last7,871°;2/11. Каждый best выбран
и frozen до собственного test; native validation точно воспроизведена,
parent SHA и strict reload проверены. Новый default не выбирали по test.

Matched low-LR update: exact KL1-го шага0,00458 вместо0,04436 при прежнем LR;
после80 принятых steps0,00564 на rollout и0,02384 на query-grid.
Уменьшился разовый policy drift; гарантии сохранения всех целей нет.

```bash
env PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/plot_arm_hypotheses.py
.venv/bin/python3 scripts/demo_arm_preacher.py --controller runs/arm_hypotheses/lr_1e4_none/best.zip --seconds 16 --output runs/arm_hypotheses/lr_1e4_none/demo_best.jsonl
.venv/bin/python3 -m py_compile scripts/check_arm_hypotheses.py scripts/check_arm_exploration_duration.py scripts/check_arm_update_drift.py scripts/resume_arm_budget_check.py scripts/plot_arm_hypotheses.py
git diff --check
```

Все exit0. График `runs/arm_hypotheses/hypotheses.png` осмотрен, SVG рядом.
Headless LR-demo16 s без failure:full RMSE22,641°, tail по75/20/105/45°
3,300/0,345/3,243/1,978°.800 последовательных v2-кадров, один episode,
20-ms timestep без reset; final minimum capacity proxy0,9648. Отдельные test
всё ещё не освоены, release не обучен; GUI не открывался. Все новые weights,
snapshots, rollout и графики остались в ignored runs; Untitled.ipynb не менялся.

## Диагностика переноса PPO со стенда на руку — 2026-11-08

Отчёт: [arm-learning-bottlenecks.md](arm-learning-bottlenecks.md).
Физика, reward, наблюдения, checkpoint contracts и веса не менялись;
новые файлы — только read-only diagnostics и документация. Новое обучение
и дополнительный полный pytest не запускались; прежний результат168 passed
из раздела ниже относится к предыдущему изменению.

```bash
.venv/bin/python3 scripts/audit_arm_learning_bottlenecks.py
.venv/bin/python3 scripts/audit_arm_execution_gap.py
.venv/bin/python3 scripts/audit_arm_vertical_transfer.py
.venv/bin/python3 scripts/audit_arm_learning_bottlenecks.py --output runs/arm_bottlenecks/audit_verified.json
env PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/audit_arm_execution_gap.py --plot-only
.venv/bin/python3 -m py_compile scripts/audit_arm_learning_bottlenecks.py scripts/audit_arm_execution_gap.py scripts/audit_arm_vertical_transfer.py
git diff --check
```

Все exit0. Первый audit86,08 s, повтор90,03 s; результаты45 исполнений,
24 ветвей/контролей,63 целевых проб и5 initial vertical эпизодов совпали точно.
Добавленные assert проверяют trigger1/нулевой excess cost, общий q/v в1 s
у21 native/импульсной ветви и telescoping identity приλ1. Совместно с двумя
остальными audits —82 полных arm эпизода,35 vertical и3 коротких arm прохода40 ms,
без failures; повтор первого набора учитывается отдельно. Строгие загрузки,
неизменность model tensors и SHA checkpoints проверены. Все генерации в ignored
`runs/arm_bottlenecks/`; `Untitled.ipynb` не изменён.

Frozen gain0,25 best20→75°: mean tail RMSE53,00°, stochastic14,08–16,26°
в4 диагностических seeds; это разные samples, не новые обучения. Native mean
останавливается около22°;200-ms усиление сгибателей на0,1 после1 s переводит
к84,33° и RMSE9,33°. Из6 стартов20–30° при цели75° итог≈22°, из35/45°≈84,36°.
На других заданиях stochastic также может быть хуже deterministic, успешной
универсальной policy не заявляем. Вертикальные successful checkpoints тоже
имеют mean/sample разрыв, чаще с лучшим mean исполнением на проверенных задачах.

У vertical reset балансирует нагрузку с residual force<1e−8 N; у arm03-start
остаток без constraints−3,02/−5,20/−3,40 Nm на20/75/125°. Новая команда не влияет
на падение в первые40 ms. В offline реконструкции5 реальных последовательностей
закон активации15/50 ms+FIFO40 ms совпал с native activation до6,2e−16;
симметричный15/15-ms фильтр применён только к записи, механика руки не менялась.
У18 парных импульсов Δreturn и ΔGAE положительны; это замороженный critic-probe
на изменённых траекториях, не повтор исторических on-policy advantages.
График `execution_gap.png` осмотрен; SVG сохранён рядом. GUI не открывался.

## Чтение h и глубина после concat — 2026-11-08

Протокол: [arm-predictor-readout-and-depth.md](arm-predictor-readout-and-depth.md).

```bash
.venv/bin/python3 scripts/audit_arm_scale_equivalence.py
.venv/bin/python3 scripts/audit_arm_scale_equivalence.py --output runs/arm_conditioning/scale_equivalence_moments.json
.venv/bin/python3 -m pytest -q tests/test_arm_depth.py tests/test_arm_conditioning.py
taskset -c 8,9 .venv/bin/python3 -m pytest -q
.venv/bin/python3 -m py_compile src/myo/arm_depth.py src/myo/train_arm_depth.py src/myo/train_arm_conditioning.py scripts/report_arm_depth.py scripts/demo_arm_preacher.py scripts/audit_arm_scale_equivalence.py
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 -m myo.train_arm_depth train
```

Scale audit exit0: на5400 состояниях `h*0,25` и соответствующий блок weights*4
сохранили prediction20, action parameters12 и V **с max abs difference0,0**.
Проверены initial/best gain1 и gain0,25; сохранённые веса не менялись.
Это проверка функциональной эквивалентности, не успешности обучения readout.
Дополнительный moments audit exit0: RMS вклада h=2,401, centered RMS1,806;
private0,486/0,199. После gain0,25 initial centered ratio≈2,3 вместо≈9,1.
Данные относятся к общему5400-state batch, не к каждому распределению новых PPO.

Целевые: **2 passed,2,76 s**. Полный набор с прежним локальным UDP-тестом:
**168 passed,6 прежних warnings,91,96 s**. Защищены включение новых слоёв в
optimizer, реальные изменения новых actor/critic weights после PPO-update,
неизменность predictor, совпадение общих исходных weights/RNG с двухслойной
веткой, восстановление RMSprop и отказ при несовместимой depth identity.
Compile exit0. Новый PPO завершил **131072 перехода,3662 optimizer steps,
106/128 KL-stop,1350,08 s**, stop maximum_transitions. С контролем совпали
все общие исходные tensors, Torch RNG, predictor/PPO/physical identities.
Изменились только post-concat слои и их depth identity; дополнительных
обучаемых параметров66048, всего PPO238989.

```bash
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 -m myo.train_arm_depth evaluate
taskset -c 4 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 scripts/demo_arm_preacher.py --controller runs/arm_depth/best.zip --seconds 16 --output runs/arm_depth/demo_best.jsonl
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_arm_depth.py --plot --audit-features
```

Все exit0. Best98304: validation28,066°;1/9 (105→105°), test **33,856°;1/11**
(15→15°). Last test30,865°;2/11 (15→15°,120→120°). Оба test без failures;
best selection зафиксирован до test, predictor encoder/head/normalization frozen
проверены побитово, native validation воспроизведён. Нового best по test нет.
Контроль2×128:10,400°;2/11; увеличение глубины при прежнем PPO не улучшило результат.
Непрерывное демо16 s без failure:full RMSE55,552°,capacity proxy0,9791,
800 последовательных v2-кадров без reset. Проверены timestamps/sequence;
полезного непрерывного движения пока нет. Feature audit на прежних5400 состояниях:
mean-|Δu|0,03103/0,04108 для mean/shuffle h, веса не менялись.
График `runs/arm_depth/depth.png` осмотрен; `.svg`, JSON-таблицы и traces
сохранены рядом. GUI/другие seeds/backends не проверялись.

## Learning rate и масштаб h1 MyoArm — 2026-11-08

Протокол: [arm-ppo-conditioning.md](arm-ppo-conditioning.md).
Старые core-модули с закреплёнными hashes и checkpoints не переписаны.

```bash
.venv/bin/python3 -m pytest -q tests/test_arm_conditioning.py
taskset -c 8,9 .venv/bin/python3 -m pytest -q
.venv/bin/python3 -m py_compile src/myo/arm_conditioning.py src/myo/train_arm_conditioning.py scripts/demo_arm_preacher.py
.venv/bin/python3 scripts/demo_arm_preacher.py --controller runs/arm_conditioning/gain_025/initial.zip --seconds 0.2
```

Целевой: **1 passed,2,13 s**. Полный набор, с разрешённым прежним локальным
UDP-тестом: **167 passed,6 прежних warnings,125,73 s** одновременно с тремя
новыми PPO на отдельных CPU. Проверены одинаковые trainable weights,
тождественность gain1 исходной policy, точное применение gain0,25, заморозка
предиктора после PPO-update, сохранение/загрузка масштаба и отказ при расхождении
gain с metadata. Compile exit0. Начальный headless demo0,2 s без failure;
это проверка загрузки/исполнения, не качества управления.

```bash
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 -m myo.train_arm_conditioning train --trial lr_1e4
taskset -c 4 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 -m myo.train_arm_conditioning train --trial lr_3e5
taskset -c 6 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 -m myo.train_arm_conditioning train --trial gain_025
```

Обучение завершено: lr1e−4 **131072 перехода,10227 optimizer steps,1640,1 s**,
1/128 KL-stop; lr3e−5 **65536,5120,965,8 s**,0/64 KL-stop, stop validation
plateau по прежнему критерию; gain0,25 **131072,5829,1624,3 s**,85/128 KL-stop.
Остальные остановлены по бюджету, не по сходимости. Запуски выполнялись
одновременно на отдельных CPU; wall-clock не является чистым CPU-бенчмарком.

```bash
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 -m myo.train_arm_conditioning evaluate
taskset -c 0 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 scripts/demo_arm_preacher.py --controller runs/arm_conditioning/lr_1e4/best.zip --seconds 16 --output runs/arm_conditioning/lr_1e4/demo_best.jsonl
taskset -c 4 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 scripts/demo_arm_preacher.py --controller runs/arm_conditioning/lr_3e5/best.zip --seconds 16 --output runs/arm_conditioning/lr_3e5/demo_best.jsonl
taskset -c 6 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 scripts/demo_arm_preacher.py --controller runs/arm_conditioning/gain_025/best.zip --seconds 16 --output runs/arm_conditioning/gain_025/demo_best.jsonl
taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_arm_conditioning.py --plot --audit-features
```

Все exit0. Selections зафиксированы до test, native validation воспроизведён,
проверены strict loading и frozen encoder/head/mean/std, исходные trainable weights
и Torch RNG. Test best: lr1e−4 **32,547°;0/11**, lr3e−5 **56,944°;1/11**
(только15→15°), gain0,25 **10,400°;2/11** (75→20°,25→105°).
Последний вариант улучшил исходный frozen29,643°;0/11, но не baseline без h1
7,334°;2/11. Все непрерывные демо16 s без failure, по800 v2-кадров без reset,
последовательные timestamps/sequence; ошибки52,453/55,944/50,430° соответственно.
Хорошего непрерывного управления пока нет. График `conditioning.png` осмотрен;
JSON-проверки, таблицы по параметрам/сценариям и проверки использования/масштабов
ветвей сохранены в ignored `runs/arm_conditioning/`. Нового test-бенчмарка,
проверки других seeds, GPU или GUI этот опыт не устанавливает.

## Использование признаков предиктора руки — 2026-11-08

```bash
.venv/bin/python3 -m py_compile scripts/audit_arm_predictor_features.py
.venv/bin/python3 scripts/audit_arm_predictor_features.py
```

Оба exit0. Без обучения: 5400 одинаковых состояний, best/last и none-control;
mean/shuffle/zero подмены только h1, raw-входы сохранены. Проверены совпадение
ручной композиции actor/critic с production policy, нулевая реакция none-control,
неизменность weights/buffers/hash. Native validation воспроизведён с atol1e−5°.
54 headless-эпизода (10800 переходов),0 physical failures: best original/mean/zero
11,503/43,359/19,821° и3/0/1 pass; last13,849/9,152/12,131° и0/3/0 pass.
Полный отчёт — [arm-balanced-predictor.md](arm-balanced-predictor.md#проверка-использования-признаков--2026-11-08),
числа/идентичности/результаты по сценариям — ignored
`runs/arm_preacher_balanced/feature_use_audit.json`.
Признаки используются, польза зависит от checkpoint. Test, selection, физика,
reward и checkpoint contracts не менялись; полный pytest для этого отдельного
диагностического скрипта не повторялся.

## Расширенный набор предиктора руки — 2026-11-08

Протокол: [arm-balanced-predictor.md](arm-balanced-predictor.md).
Добавлены независимые capacity/амплитуды, feedback-пробы удержания/движения,
быстрые возмущения и causal sampling по состояниям. Исходные physical/observation/
reward/checkpoint модули и старые weights не менялись.

```bash
taskset -c 4 .venv/bin/python3 -m pytest -q tests/test_arm_balanced.py
taskset -c 6,7 .venv/bin/python3 -m pytest -q
taskset -c 2,4,8 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python3 -m myo.train_arm_balanced collect --workers 3
.venv/bin/python3 -m py_compile src/myo/arm_balanced.py src/myo/train_arm_balanced.py scripts/report_arm_balanced.py
```

Целевые: **2 passed,1,83 s**; полный pytest вне sandbox для прежнего локального
UDP-теста: **166 passed,6 прежних warnings,52,58 s**. Проверены реальное
удержание 80° без фиксации сустава, полное произведение capacity/предела возбуждения
и повышение веса редких средних положений без удаления распространённых.
Compile: exit0. Сбор завершён: **621 эпизод,124200 переходов,0 failures**,
67400/27800/29000 train/validation/test,285,08 s wall-clock на 3 локальных workers.
Data audit подтвердил конечность массивов и побитовую непрерывность физических
состояний/FIFO внутри эпизода; изменение внешней цели не считается reset.
Train30–120°:39,54% вместо 10,56%; middle/slow6939 кадров вместо 59,
максимальная непрерывная последовательность 3,64 s вместо 0,08 s.

```bash
taskset -c 6,7 .venv/bin/python3 -m myo.train_arm_balanced predictor --threads 2
taskset -c 4 env PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_arm_balanced.py
taskset -c 4 .venv/bin/python3 -m myo.train_arm_balanced train
```

Predictor exit0: **78000 дополнительных updates**, best66000, stop validation
plateau,143,58 s wall-clock. Новый общий test q20/v20:0,0541°/2,746°/s
против старых 0,2148°/7,499°/s на тех же эпизодах. На новых policy-test прогонах:
**0,0748°/3,452°/s** против **0,2891°/10,390°/s**; q60 **0,2124°** против 0,9416°.
Прежний motor-test не ухудшился: q20 0,0459° против 0,0488°.

Initial predictor совпадает с прежним best побитово; mean/STD сохранены.
Новый PPO initial содержит точно selected encoder/head/normalization.
Private weights, initial Torch RNG, PPO settings и task contract совпали с
прежними none/frozen. Проверки exit0, `initialization_audit.json`:passed.
PPO завершён: **131072 перехода,1122,50 s wall-clock,4107 actual optimizer steps**,
stop maximum_transitions, не convergence. Best122880: validation11,503°;3/9,
все три прохода для цели 75°. Test best **29,643°;0/11**; last **21,430°;1/11**.
Test не использован для изменения selection. Улучшения управления против старого
baseline не получено; выявлен перевес цели 75° в validation-критерии.

```bash
taskset -c 4 .venv/bin/python3 -m myo.train_arm_balanced evaluate
taskset -c 2 .venv/bin/python3 scripts/demo_arm_preacher.py --controller runs/arm_preacher_balanced/frozen/best.zip --seconds 16 --output runs/arm_preacher_balanced/demo_best.jsonl
taskset -c 6 .venv/bin/python3 scripts/demo_arm_preacher.py --controller runs/arm_preacher_balanced/frozen/last.zip --seconds 16 --output runs/arm_preacher_balanced/demo_last.jsonl
taskset -c 4 .venv/bin/python3 scripts/probe_balanced_arm_policy.py
taskset -c 6 env PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_arm_balanced.py
```

Все команды exit0. `evaluation_audit.json`:passed —3 strict loads, точное
воспроизведение selected validation, правильный ранг, frozen predictor weights
и нормировка совпадают побитово от initial до last; настройки/инициализация
совпали с предыдущими запусками. На 5400 новых stochastic-состояниях выбранного
контроллера новый/старый предикторы: q20 **0,0844/0,3685°**, v20 **3,186/12,578°/s**,
q60 **0,2420/1,2028°**;0 failures, эти данные не дообучают предиктор.

Headless demo best/last:16 s, без termination/reset, full RMSE51,753/37,937°,
final min capacity proxy0,9605/0,9704. Проверены по 800 v2 кадров, непрерывные
20-ms timestamps и один episode. Release/protection и интерактивный GUI не проверялись.
Графики comparison/movement сгенерированы и просмотрены. После полного pytest
семантика core-модулей не менялась; добавлены только отчёты/диагностические scripts
и уточнён комментарий к pending-команде в новом sampler.

## Покрытие motor-пробами состояний PPO руки — 2026-11-07

Отчёт: [arm-predictor-movement-coverage.md](arm-predictor-movement-coverage.md).
Повторного обучения нет. Физика, наблюдения, награда, serialization и веса
не менялись; новый скрипт только собирает и анализирует диагностические траектории.

```bash
taskset -c 4 env PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/compare_arm_movement_coverage.py
taskset -c 4 env PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/compare_arm_movement_coverage.py --reuse
.venv/bin/python3 -m py_compile scripts/compare_arm_movement_coverage.py
```

Сбор/анализ exit0. Исходный motor-train 12800 кадров; новые stochastic-срезы
initial/none-best/frozen-best по 27 TRAIN_CASES, по 5400 кадров, по 0 failures.
Это новые прогоны, не сохранённые исторические rollout обучения PPO.
Строгие loader сверили source/config/checkpoint identities; `--reuse` проверил
совпадение выбранных checkpoint с записанными identities.

Motor train: ниже 30°81,62%, в 30–120°10,56%; средняя область с |v|≤5°/s:
59 кадров, максимальная последовательность 80 ms, ни одной≥500 ms.
PPO none/frozen: в 30–120°57,74/57,26%; median |Δu| за 20 ms 0,05807/0,05778
против 0,00163 в motor train. Подтверждена связка initial capacity/limit в исходных
эпизодах: (1,.06),(0.8,.12),(1,.24),(0.8,.4), по 16 train-эпизодов.

Предиктор неизменен. Motor-test q20/v20 повторены: 0,04877°/3,536°/s;
новые PPO none/frozen: 0,3061/0,3262° и 10,403/10,957°/s. Ошибка остаётся выше
внутри общей области 30–120°, поэтому разница углов объясняет сдвиг не полностью.
Сгенерированы и просмотрены distributions/trajectories; полный JSON и npz в
ignored `runs/arm_preacher/movement_coverage/`. Pytest не повторялся: core-код
со времени полного 164-passed прогона не менялся.

Дополнительные локальные Python-проверки: exit0. Сверены все 27 case и config
identities, 16200 конечных допустимых переходов, побитовая непрерывность соседних
кадров внутри эпизода, по5346 окон горизонта3 без переходов через reset,
совпадение motor-test метрик с исходным training.json с допуском1e−5,
конечность массивов и относительные ссылки документации. `git diff --check`: exit0.

## Возобновление обучения MyoArm на скамье Скотта — 2026-11-07

Протокол: [arm-preacher-training.md](arm-preacher-training.md).
Фактические команды:

```bash
taskset -c 6 .venv/bin/python3 -m pytest -q tests/test_arm_precision.py
taskset -c 6,7 .venv/bin/python3 -m pytest -q
taskset -c 2 .venv/bin/python3 -m myo.train_arm_precision collect
taskset -c 6,8 .venv/bin/python3 -m myo.train_arm_precision predictor --threads 2
taskset -c 4 .venv/bin/python3 -m myo.train_arm_precision train --trial none
```

Целевые: **3 passed, 2,32 s**. Полный pytest вне sandbox для прежнего UDP-теста:
**164 passed, 6 прежних warnings, 54,70 s**. Native-переходы нового wrapper побитово
совпадают с исходной средой; моторная очередь сжимается без потери порядка.
Проверены причинность предиктора, split по эпизодам, frozen weights, embedded
identity и следующий одинаковый RMSprop-step после загрузки.

Feasibility: 8 нативных проходов эвристики по 4 s, без физической помощи.
20→75 и 75→20: steady RMSE 0,415/0,437°;90→135:0,371°.
90→150 остановился на 141,722° с contact peak 16,306 N; не объявлен достижимым.
`heuristic_feasibility.json` и его числа приведены в протоколе.

Collection exit0: **112 эпизодов,22400 переходов,0 failures**,64/24/24 эпизода
по train/validation/test;12800/4800/4800 переходов. Сбор 168,56 s wall-clock
вместе с параллельными тестами/обучением, не изолированный performance benchmark.
Predictor завершён: **27500 updates**, stop по заданному validation plateau,
best 21500,70,41 s wall-clock. Held-out test: q20/q60 RMSE **0,04877/0,22094°**,
v20/v60 **3,536/5,705°/s**. Persistence: q20/q60 **0,9844/2,9315°**,
v20/v60 **10,363/24,861°/s**. Train-данные не используют эвристику или цели.
Запущена frozen-ветвь:

```bash
taskset -c 2 .venv/bin/python3 -m myo.train_arm_precision train --trial frozen
```

Initial private weights и начальное состояние Torch RNG совпали у none/frozen;
предиктор не переобучается по их PPO-rollout. Обе PPO-ветви завершены по бюджету
**131072 перехода**, не по плато: none 1404,61 s, frozen 1398,06 s wall-clock
(частично параллельные процессы). Actual optimizer steps 6074/5729.

```bash
taskset -c 4 .venv/bin/python3 -m myo.train_arm_precision evaluate
taskset -c 2 .venv/bin/python3 scripts/demo_arm_preacher.py --controller runs/arm_preacher/none/best.zip --seconds 16 --output runs/arm_preacher/demo_none.jsonl
taskset -c 6 .venv/bin/python3 scripts/demo_arm_preacher.py --controller runs/arm_preacher/frozen/best.zip --seconds 16 --output runs/arm_preacher/demo_frozen.jsonl
taskset -c 4 env PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_arm_preacher.py
.venv/bin/python3 -m py_compile scripts/demo_arm_preacher.py scripts/report_arm_preacher.py
```

Все перечисленные прогоны exit0. Best выбран по validation до открытия test:
none 131072, frozen 90112. Validation **8,374°;2/9** против **29,408°;1/9**;
test best **7,334°;2/11** против **28,085°;1/11**; test last 7,334°;2/11
против 27,845°;1/11. Знакомые опорные пары выделены в отчёте; без них новые
test-пары: **7,368°;2/9** против **27,659°;0/9**. Успех none —75→20° и 25→105°;
frozen прошёл только знакомое 15→15°. Независимая эвристика прошла 9/9 validation,
средний tail RMSE 0,215°, не используется как teacher.

`evaluation_audit.json`: **passed**, 6 строгих загрузок initial/best/last,
побитово неизменные predictor encoder/head/normalization у обеих ветвей,
точное воспроизведение selected validation и проверка его ранга.
16-s demo headless: full RMSE 34,832/56,263°, без termination, final min capacity
proxy 0,9702/0,9768. По 800 v2-пакетов, sequence 0…799, один episode, непрерывные
20-ms timestamps. Это не проверка release/protection и не успешная приёмка D1.

Графики comparison/movement сгенерированы и просмотрены. EGL offscreen через
существующий `myo.demo.render_frame`: none 35→95°, конец 97,45°, кадр
`runs/arm_preacher/none_validation_pose.png` просмотрен; интерактивный GUI
не запускался. После полного pytest менялись только helper scripts/документация;
код динамики/награды/наблюдений/сериализации с того момента не менялся.

## План параллельной работы десктоп/ноутбук — 2026-11-07

Добавлен [план Unreal-демо скамьи Скотта](desktop-unreal-preacher-plan.md).
Проверены текущие MyoArm importer, telemetry v2 и поведение CLI; учтены
отсутствующие UE rig exporter, Windows bootstrap и непрерывный headless bridge.
Flesh curve override и surface binding сверены с официальными документацией
и tutorial Epic; источники стоят возле соответствующих пунктов плана.
Там же отделены существующие возможности Python от будущих desktop-проверок.

Изменена только документация. Проверены относительные Markdown-ссылки,
`git diff --check` и `git diff --cached --check`: exit0.
Новые обучения, установка UE, импорт анатомических assets, Flesh-симуляции
и Windows-команды в этой работе не выполнялись; pytest не повторялся.

## Фиксация кода и результатов перед публикацией — 2026-11-07

Перед общим коммитом MyoArm, вертикальных стендов и исследовательских отчётов
повторно выполнены:

```bash
taskset -c 6,7 .venv/bin/python3 -m pytest -q
.venv/bin/python3 -m compileall -q src/myo scripts
git diff --check
git diff --cached --check
```

Полный pytest вне sandbox для существующего UDP-теста: **161 passed,
6 прежних warnings, 36,50 s**. Compileall и проверка whitespace: exit0.
Проверка относительных Markdown-ссылок в README и docs локальным Python:
отсутствующих файлов нет. Результаты опытов сохранены в отчётах и карте
экспериментов; новых обучений при фиксации не запускали.
Сгенерированные данные и новые weights остаются в ignored `runs/`;
ранее разрешённый tracked demo checkpoint не изменён. Лог MuJoCo и Jupyter
checkpoint-кэш игнорируются, несвязанный `Untitled.ipynb` оставлен локально.

## Сопоставление PPO с предиктором и без — 2026-11-07

Новых обучений и физических оценок не запускали. Существующие `summary.json`
из `shared_vertical` (3), `detached_vertical/online`, `pretrained_vertical` (3)
и `vertical_precision/combined` проверены локальным `.venv/bin/python3`:
assert одинаковых physical contract, observation_names, reward, train/validation/
test cases, seed0 и steps131072; все8 имеют status=complete. Проверка exit0.
Из сохранённых test-score пересчитаны относительные изменения RMSE:
fixed pretrained против strict independent **−61,4017%**, против исторического
precision PPO **−10,7812%**. Эти сравнения отличаются архитектурой/затратами,
а историческое — также KL-контролем; это не новые изолированные абляции.
Полная таблица и ограничения добавлены в
[pretrained-predictor-ppo.md](pretrained-predictor-ppo.md#сравнение-с-ppo-без-предиктора).
Изменена только документация; `git diff --check` exit0. Pytest не повторялся.

## PPO с отдельно предобученным h1 — 2026-11-07

Протокол: [pretrained-predictor-ppo.md](pretrained-predictor-ppo.md).
Проверены перенос только predictor weights/train-нормировки в новую policy,
одинаковая private-инициализация трёх ветвей, frozen/online поведение и runtime
без внешнего `.pt`. В policy сохранены оба RMSprop states; одинаковый следующий
шаг после загрузки даёт побитово те же веса, подмена predictor identity отвергается.

```bash
.venv/bin/python3 -m pytest -q tests/test_pretrained_vertical.py
taskset -c 6,7 .venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo.train_pretrained_vertical prepare
taskset -c 2 .venv/bin/python3 -m myo.train_pretrained_vertical train --trial random_frozen
taskset -c 3 .venv/bin/python3 -m myo.train_pretrained_vertical train --trial pretrained_frozen
taskset -c 4 .venv/bin/python3 -m myo.train_pretrained_vertical train --trial pretrained_online
```

Целевой тест: **1 passed, 1,88 s**. Полный pytest вне sandbox для прежнего
UDP-теста: **161 passed, 6 прежних warnings, 44,43 s**. Prepare exit0.
Три обучения завершены, все exit0: по131072 PPO-перехода, seed0. Затем:

```bash
taskset -c 6 .venv/bin/python3 scripts/evaluate_pretrained_vertical.py
taskset -c 2 .venv/bin/python3 scripts/audit_vertical_precision_critic.py \
  --controller runs/pretrained_vertical/random_frozen/best.zip \
  --state-controller runs/vertical_precision/combined/best.zip \
  --output runs/pretrained_vertical/random_frozen/critic_audit.json
taskset -c 4 .venv/bin/python3 scripts/audit_vertical_precision_critic.py \
  --controller runs/pretrained_vertical/pretrained_frozen/best.zip \
  --state-controller runs/vertical_precision/combined/best.zip \
  --output runs/pretrained_vertical/pretrained_frozen/critic_audit.json
taskset -c 8 .venv/bin/python3 scripts/audit_vertical_precision_critic.py \
  --controller runs/pretrained_vertical/pretrained_online/best.zip \
  --state-controller runs/vertical_precision/combined/best.zip \
  --output runs/pretrained_vertical/pretrained_online/critic_audit.json
taskset -c 10 .venv/bin/python3 scripts/demo_shared_vertical.py \
  --controller runs/pretrained_vertical/pretrained_frozen/best.zip --seconds 12
taskset -c 6 env PYTHONPATH=/tmp/myo-supervised-plot-deps \
  MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/report_pretrained_vertical.py
```

Evaluator exit0, `evaluation_audit.json`: **passed**,9 strict loads,
33 точных native test-replay. У pretrained-пары совпали initial weights и
первая rollout; у всех одинаковы private weights. Оба frozen predictor/head
побитово неизменны, все delta-нормировки сохранены. Best выбран до test.
Контроль повторил всю validation-кривую, первую rollout и test C12 frozen.

Best test random/frozen/online: **3/11;8,320 mm /10/11;3,187 mm /1/11;19,871 mm**,
0 physical failures. Last: **3/11;9,354 /5/11;4,303 /1/11;19,871 mm**.
Frozen best106496: все4 test-движения установились0,689–1,402 s,
максимальный перелёт2,450 mm. Hold+70 mm не прошёл (RMSE8,661 mm).
Online на широком motor test ухудшил v20 с2,530 до60,923 mm/s, но на одинаковых
старых управляющих validation улучшил7,526→3,767 mm/s: специализация/забывание,
не глобальное ухудшение predictor. Guard всех PPO-принятых шагов≤.03;
двухфазный online KL до.031958 (выше.03 дважды), predictor-only до.005182.

MC-аудиты exit0: одинаковые12 состояний,6 stochastic продолжений по16 s,
gamma .99, reward×.1, без bootstrap. RMSE V к MC mean: **.31458 /.42424 /.80431**,
bias **−.00598 /−.15820 /+.20146**, EV **.1681 /.3678 /.2227**,0 failures.
Source-состояния SHA256 `48d616b9263e4ac4ab677d0d435ddce10b284b3230a2462b9861d50d1050596e`.
Это малая диагностика, различия critic сами по себе не объясняют policy.

Headless demo exit0:12 s,0 failures, full RMSE18,003 mm с переходами целей.
В повторном последовательном12-s replay без reset последние секунды участков
+60/−60/0 mm дали RMSE **4,304 /1,176 /1,830 mm**, сохранены `demo_trace.json`
и `demo_evaluation.json`. GUI в этой сессии не открывалась.
Reporter exit0, экспортированы `comparison.png/svg` и `movement.png/svg`;
оба PNG просмотрены, подписи/кривые читаемы. `py_compile` новых модулей,
evaluator/reporter и изменённых demo/critic-loader прошёл; `git diff --check`
exit0. Checkpoints и metadata подтверждены ignored через `git check-ignore`.

## Отдельный предиктор на разнообразных моторных пробах — 2026-11-07

Протокол: [body-predictor-pretraining.md](body-predictor-pretraining.md).
Сохранены сеть 8→4×128 SiLU→6 (51462 параметра), текущие 8 входов и RMSprop.
Actor/critic не обучаются. Полная механика 4 без изменений; фиксированные
whole-episode splits, нормировка по полным train-переходам.

Фактические команды:

```bash
.venv/bin/python3 -m pytest -q tests/test_body_predictor.py
taskset -c 2,3 .venv/bin/python3 -m pytest -q
taskset -c 0 .venv/bin/python3 -m myo.train_body_predictor collect
taskset -c 2,3 .venv/bin/python3 -m myo.train_body_predictor train
```

Целевые: **3 passed, 2,07 s**. Полный pytest вне sandbox для существующего
UDP-теста: **160 passed, 6 прежних warnings, 41,90 s**. Проверены native labels,
границы эпизодов, детерминизм/плавность команд, исходная инициализация сети,
замороженная train-нормировка, экспорт без изменения private heads,
causal FIFO и побитово одинаковый следующий RMSprop update после загрузки.

Collection exit0: **56 эпизодов, 33600 полных переходов, 0 physical failures**.
Train/validation/test: **32/12/12** эпизодов, **19200/7200/7200** переходов.
Положение −0,19446…+0,15031 m, скорость −1,5363…+1,0015 m/s на 20-ms кадрах.
Median корреляции команд внутри train-эпизодов independent/reciprocal/common/
multisine: **−0,0123 / −0,8263 / +0,9596 / +0,0672**. Сигналы заранее
генерируются, реакции на положение, силовой поддержки и изменения массы нет.
Train-данные зафиксированы и повторяются; test до выбора не оценивался.
Обучение и итоговый аудит завершены. После фиксации выбора выполнены:

```bash
taskset -c 2,3 .venv/bin/python3 -m myo.train_body_predictor evaluate
taskset -c 0 env PYTHONPATH=/tmp/myo-supervised-plot-deps \
  MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/report_body_predictor.py
```

Все exit0. Обучение **50000 updates**, **6,4 млн** предъявлений с повторами,
**100,5 s** CPU/2 torch threads. Best **44500** по всей six-channel validation
MSE **0,005118**; train на нём **0,004489**. Лимит 50000 наступил раньше
plateau-stop: прошло 5500 updates после последнего значимого улучшения,
patience6000. После 10000 validation MSE была 0,02122; плато на том бюджете нет.

Новый test best q RMSE **0,0409/0,1848/0,4146 mm**, v RMSE
**2,530/4,557/4,951 mm/s** через 20/60/120 ms. Прежний parallel actor-best
на тех же 7200 началах через 20 ms: **0,9998 mm /101,160 mm/s**.
Одношаговые ошибки относительно RMS изменения q/v **1,04% /8,94%**.
На старой fixed-policy validation v60 **37,23→10,07 mm/s**.
Last чуть точнее по q/v, но хуже по совокупной test MSE: **0,006174**
против best **0,005224**, поэтому после test выбор не менялся.

На **12** новых общих native-пробах 60 ms replay источников точный,
Jacobian relative error best **61,04%**, parallel actor-best **30,12%**;
cosine best **0,9905**, проекция на native **1,5704**. При лучшем прогнозе
величина реакции на вмешательство остаётся неточной. Pending/0 и activation
в train коррелируют **0,9895/0,9910**; причинная роль этой связи не установлена.

`final_audit.json` — passed:3 строгих загрузки, train-нормировки точные,
whole-episode окна 20/40/60/120 ms без reset-переходов, best/plateau критерий
пересчитан. `learning_and_forecast` и `motor_examples` экспортированы PNG/SVG,
PNG просмотрены. `py_compile` новых модулей и `git diff --check` прошли.
Веса/данные остались ignored; новое PPO и GUI не запускались.

## Раздельное обучение признаков предиктора и PPO — 2026-11-07

Протокол: [detached-predictor-ppo.md](detached-predictor-ppo.md).
Новые independent-private actor/critic ветви используют detached h1 предиктора;
PPO и predictor имеют раздельные RMSprop. Предиктор обновляется после PPO,
его собственный вклад в сдвиг policy KL измеряется отдельно и не ограничен
PPO-guard. Физика/награда/13 наблюдений сохранены.

Фактические проверки:

```bash
.venv/bin/python3 -m pytest -q tests/test_detached_vertical.py
taskset -c 0,1 .venv/bin/python3 -m pytest -q
```

Целевые: **3 passed, 2,52 s**. Полный набор вне sandbox для существующего
локального UDP-теста: **157 passed, 6 прежних warnings, 40,40 s**.
Проверены stop_gradient и отсутствие чужих optimizer-параметров, causal FIFO,
неизменность actor/critic параметров при predictor-update при ненулевом policy
KL, frozen-контроль, заморозка train-нормировки и побитово одинаковый следующий
шаг обоих RMSprop после строгой загрузки. Source/metadata хранят новый контракт.

Подготовлена копия прежних 1800 predictor-validation transitions с проверкой SHA;
их не используют в optimizer/normalizer/выборе best. Завершены свежие matched
frozen/online по 131072 transitions, seed0; всего 262144 перехода.

```bash
taskset -c 0 .venv/bin/python3 -m myo.train_detached_vertical prepare
taskset -c 2 .venv/bin/python3 -m myo.train_detached_vertical train --trial frozen
taskset -c 3 .venv/bin/python3 -m myo.train_detached_vertical train --trial online
```

Оба обучения exit 0, остановка по бюджету. Начальные веса и первые 1024
observations/actions/labels/valid mask совпали по SHA. Frozen h1 и обе train-only
нормировки проверены после обучения. Params обеих ветвей 187531, из них
predictor 51462. Best frozen/online: **90112 / 114688**; test selections
зафиксированы до test, модели и гиперпараметры по test не менялись.

Итоговые фактические команды:

```bash
taskset -c 2 .venv/bin/python3 scripts/evaluate_detached_vertical.py
taskset -c 3 .venv/bin/python3 scripts/audit_vertical_precision_critic.py \
  --controller runs/detached_vertical/frozen/best.zip \
  --state-controller runs/vertical_precision/combined/best.zip \
  --output runs/detached_vertical/frozen/critic_audit.json
taskset -c 4 .venv/bin/python3 scripts/audit_vertical_precision_critic.py \
  --controller runs/detached_vertical/online/best.zip \
  --state-controller runs/vertical_precision/combined/best.zip \
  --output runs/detached_vertical/online/critic_audit.json
taskset -c 0 env PYTHONPATH=/tmp/myo-supervised-plot-deps \
  MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/report_detached_vertical.py
taskset -c 0 .venv/bin/python3 scripts/demo_shared_vertical.py \
  --controller runs/detached_vertical/online/best.zip --seconds 12
```

Все exit 0. Best test: **8,320 mm; 3/11 / 4,531 mm; 8/11**. Last test:
**9,354 mm; 3/11 / 6,438 mm; 5/11**. У online все четыре движения достигают
устойчивого допуска5 mm/0,03 m/s за0,471–1,209 s, перелёт максимум2,788 mm.
Три провала best относятся к удержаниям −30/+70/−35 mm. Best/last test
без physical failure. Все22 best-траектории повторены native побитово
на границах управления, измерены физические samples1 ms. Строгая загрузка
initial/best/last обеих ветвей воспроизвела selection/validation; дополнительно
сравнены frozen h1 initial/last. Предикторная validation и normalizer сохранены.

Online q RMSE на fixed validation **0,244/1,511 mm** через20/60 ms; на
собственной **0,243/1,505 mm**. Скорость на собственной validation хуже
persistence: **18,89/37,17** против **8,61/22,95 mm/s**. Native-Jacobian
на тех же шести почти установившихся пробах: relative RMSE **16,14%**,
cosine **0,9932**. Точность всех каналов или всего пространства не заявляется.
Read-only замена h1 средним дала KL actor **0,00119 / 0,05355**; веса не менялись.

Принято PPO updates **2715 / 3196**, отвергнуто **103 / 102**, без принятого
update остались **30 / 24** из128 rollout. Предиктор online: **10240** updates.
PPO-guard соблюдён; predictor-only KL mean/max **0,001614 / 0,010002**,
суммарный KL обеих фаз максимум **0,030883**, выше0,03 один раз.
Ограничение0,03 не распространяется на отдельную supervised-фазу.

Critic MC: **144** stochastic-продолжения16 s без срыва; 12 общих snapshots,
по6 повторов, одинаковый SHA состояний и seed917. RMSE **0,315 / 0,465**,
bias **−0,006 / −0,400**, EV **0,168 / 0,192**. Обучение признаков не улучшило
critic на этой малой выборке собственных returns. Headless-демо12 s без срыва,
whole-run RMSE **16,621 mm**, включая смены целей. GUI не проверялся.
PNG просмотрен, SVG экспортирован; `py_compile` изменённых модулей и `git diff --check` прошли.
Сырые отчёты, нормировки и веса остались ignored в `runs/detached_vertical/`.

## Общий кодировщик: первое сравнение — 2026-11-07

Полный протокол и результаты: [shared-encoder-ppo.md](shared-encoder-ppo.md).
Три новые ветви с нуля: independent, shared, shared_prediction; масса 1 kg,
механика 4, прежние 33 задания и log-reward. У всех одинаковый точный
post-update KL-guard, включая critic/auxiliary-вклад; отменённый update
восстанавливает параметры и RMSprop state. Прежняя физика и checkpoints не менялись.

Фактические команды (три обучения параллельно):

```bash
.venv/bin/python3 -m pytest -q tests/test_shared_vertical.py
taskset -c 0,1 .venv/bin/python3 -m pytest -q
taskset -c 0 .venv/bin/python3 -m myo.train_shared_vertical prepare
taskset -c 2 .venv/bin/python3 -m myo.train_shared_vertical train --trial independent
taskset -c 3 .venv/bin/python3 -m myo.train_shared_vertical train --trial shared
taskset -c 4 .venv/bin/python3 -m myo.train_shared_vertical train --trial shared_prediction
```

Целевые проверки: **3 passed, 2,20 s**. Полный pytest вне sandbox для
существующего loopback UDP-теста: **154 passed, 6 прежних warnings, 35,16 s**.
Проверены три gradient-пути в encoder, строгие нули воздействия команды
до 40 ms, terminal-observation labels до reset, замороженная train-нормировка,
точный rollback weights/optimizer и следующий RMSprop update после загрузки.

Первоначальная проверка подсказала не пропускать pending/1 и предыдущую команду
в predictor-кодировщик: иначе на втором предсказанном шаге появляется путь
действия раньше transport delay. До запусков протокол уточнён: общий encoder
видит 6 физических каналов, pending/previous/goal добавляются в actor/critic
heads, predictor получает только действующее pending/0. Проверки первых двух
нулевых и третьего ненулевого производных прошли; неверная версия не обучалась.

Подготовлены 1800 фиксированных validation-переходов прежней precision policy,
в optimizer/normalization не поступают. Общие ветви имеют одинаковые начальные
тензоры и SHA первого rollout observations/actions/labels/valid mask. Новый
independent имеет побитово те же начальные веса, что старый precision baseline.
Trainable params: independent **103301**, shared/shared_prediction **119947**
(включая predictor-head, не обучаемую у контроля shared).
Завершены **393216 transitions**, по 131072 на ветку. Best steps:
independent/shared/shared_prediction **131072 / 114688 / 131072**.
Test после фиксации трёх selections: **8,257 / 6,813 / 10,093 mm**, pass
**5/11 / 5/11 / 3/11**. Все четыре test-движения у всех трёх без перелёта,
но не входят устойчиво в допуск из-за недохода. Вспомогательная ошибка
предиктора с весом 0,05 не улучшила управление. Прежний precision best
на том же test точнее; сравнение со старым опытом меняет также KL-guard.

Итоговые фактические команды:

```bash
taskset -c 2 .venv/bin/python3 scripts/evaluate_shared_vertical.py
taskset -c 3 .venv/bin/python3 scripts/audit_vertical_precision_critic.py \
  --controller runs/shared_vertical/independent/best.zip \
  --state-controller runs/vertical_precision/combined/best.zip \
  --output runs/shared_vertical/independent/critic_audit.json
taskset -c 4 .venv/bin/python3 scripts/audit_vertical_precision_critic.py \
  --controller runs/shared_vertical/shared/best.zip \
  --state-controller runs/vertical_precision/combined/best.zip \
  --output runs/shared_vertical/shared/critic_audit.json
taskset -c 5 .venv/bin/python3 scripts/audit_vertical_precision_critic.py \
  --controller runs/shared_vertical/shared_prediction/best.zip \
  --state-controller runs/vertical_precision/combined/best.zip \
  --output runs/shared_vertical/shared_prediction/critic_audit.json
taskset -c 6 .venv/bin/python3 scripts/audit_vertical_precision_mass.py \
  --controller runs/shared_vertical/independent/best.zip --masses .5 1 1.5 \
  --output runs/shared_vertical/independent/mass_audit.json
taskset -c 1 .venv/bin/python3 scripts/audit_vertical_precision_mass.py \
  --controller runs/shared_vertical/shared/best.zip --masses .5 1 1.5 \
  --output runs/shared_vertical/shared/mass_audit.json
taskset -c 2 .venv/bin/python3 scripts/audit_vertical_precision_mass.py \
  --controller runs/shared_vertical/shared_prediction/best.zip --masses .5 1 1.5 \
  --output runs/shared_vertical/shared_prediction/mass_audit.json
taskset -c 0 env PYTHONPATH=/tmp/myo-supervised-plot-deps \
  MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/report_shared_vertical.py
taskset -c 0 .venv/bin/python3 scripts/demo_shared_vertical.py --seconds 12
```

Все exit 0. Evaluator: **9** строгих загрузок, **33** побитово совпавших
native test replay. Нормировки обоих shared-вариантов совпали; суммы
accepted/rejected по всем 384 rollout совпали со счётчиками. На всех
accepted updates KL≤0,03. Полностью без принятого update остались
**37 / 4 / 1** rollout соответственно — ограничение текущего guard,
который не делает backtracking на меньший шаг. Новых параметров после
просмотра test не подбирали.

Critic: по 12 одинаковым state snapshots × 6 stochastic-продолжений
16 s, всего **216** эпизодов без падений. SHA входных состояний у всех
ветвей одинаков. RMSE по средним MC **0,521 / 0,523 / 0,483**; у auxiliary
mean bias снизился до −0,014, но существенные state-dependent ошибки остались.

Предиктор auxiliary: position RMSE **0,149/0,821 mm** на 20/60 ms,
velocity RMSE **25,40 mm/s** на 60 ms. На шести почти установившихся
validation-пробах Jacobian relative RMSE **0,0784**, cosine **0,9976**;
это другие состояния/распределение, чем у прежнего standalone predictor.

Массы 0,5/1/1,5 kg: по 33 эпизода на ветку, всего **99**, без падений.
Все три mass-аудита при 1 kg точно повторили сохранённые test control-samples.
Auxiliary pass на 0,5/1,5 kg лишь 2/11 и 1/11 — переноса не освоил.
Headless demo auxiliary best: **12 s**, без падения, whole-run RMSE
**19,832 mm** со сменами целей; GUI не проверялся. PNG/SVG экспортированы,
PNG просмотрен. `py_compile` новых/изменённых модулей и `git diff --check`
прошли. Сырые артефакты — ignored `runs/shared_vertical/`.

## Карта истории экспериментов — 2026-11-07

Документальная пауза по решению владельца; новых обучений/rollout нет.
Создана [карта опытов](experiment-history.md): 3 исторические группы V0,
13 контроллерных и 5 predictor-групп MyoArm, 10 predictor- и 10
контроллерных/диагностических групп вертикального стенда. Есть указатель
по механике, заданиям, reward/loss, входам/памяти, архитектуре и оптимизации.
Общий кодировщик actor/critic/predictor отмечен как ещё не реализованный шаг;
соседний `bnn` и замена backprop в ближайший scope не включены.
Ссылки на обзор добавлены в README и общую методику экспериментов.

Фактическая проверка inline Python через `.venv/bin/python3`: **59** локальных
ссылок/якорей без пропусков, 38 уникальных последовательных M/D/P/C-обозначений
плюс 3 V0-группы. Свежие position-RMSE, pass по массам и RMSE критика сверены
с сохранёнными `runs/vertical_budget`, `vertical_precision` JSON. Проверка
процессов `ps -eo pid,args` с фильтром по project train/continue runners
не обнаружила активных обучений. `git diff --check` — exit 0.
Изменены только Markdown-документы; pytest повторно не запускался.

## Независимая оценка критика — 2026-11-07

```bash
taskset -c 2 .venv/bin/python3 scripts/audit_vertical_precision_critic.py
.venv/bin/python3 -m py_compile scripts/audit_vertical_precision_critic.py
git diff --check
```

Все exit 0. Только frozen precision best, масса 1 kg, без обучения/отбора.
12 реальных состояний (начала/200 ms четырёх движений и четыре hold),
по шесть stochastic Beta-продолжений 16 s: 72 траектории, 57 600 transitions,
ни одного physical failure. Наблюдения после восстановления snapshot
совпали побитово. MC-return использует training reward ×0,1 и gamma=0,99,
без критика в targets и без bootstrap. Положительный непосчитанный хвост
после 16 s ограничен сверху 0,00323. Учебный timeout 4 s не прерывает
фиксированную цель, что соответствует TimeLimit-bootstrap алгоритма.

RMSE critic против средних MC **0,455**, среднее смещение **−0,409**,
EV **0,190** на выбранных 12 состояниях. Средние returns 8,067–8,834;
SE среднего по шести повторам около 0,04–0,08. Все 12 оценок занижены.
Есть пример неверного ранжирования: подъём −30→+60 на 200 ms vs hold 0.
Не доказано, что именно critic причинно объясняет провалы actor.

Логи последних 32 training rollout: медиана EV **0,263**, отрицательных
**5/32**. SB3 считает их по pre-update values и GAE/TD(lambda)-targets,
а не по независимой истинной доходности сохранённого best. Value loss
не следует интерпретировать как ошибку прогноза одиночной награды.

Отчёт: [vertical-critic-audit.md](vertical-critic-audit.md).
JSON: `runs/vertical_precision/critic_audit.json`. Динамика, reward,
serialization и веса не менялись; полный pytest для диагностического
скрипта повторно не запускался.

## Масса груза и пик верхней активации — 2026-11-07

```bash
taskset -c 2 .venv/bin/python3 scripts/audit_vertical_precision_mass.py
.venv/bin/python3 -m py_compile scripts/audit_vertical_precision_mass.py
git diff --check
```

Все exit 0. Только замороженный precision best (4×128 SiLU RMSprop),
строгая загрузка при train-массе 1 kg; дальнейшее изменение `mass_kg`
между эпизодами явно записано как evaluation-only distribution shift.
Массы 0,5/1/1,5/2/3/5 kg, по 11 эпизодов 4 s, всего 66 прогонов.
Начальный баланс сил под каждую массу: ошибка <1,5e−8 N; в ходе эпизода
масса/физика/активация не корректировались. Пики записаны с шагом 1 ms;
исходные 2200 control-отсчётов при 1 kg совпали побитово с сохранённым test.
Новых unit-тестов/обучений не добавляли: динамика, reward и loader не менялись.

При 1 kg верхняя activation достигает **33,37%** (подъём −30→+60 mm,
60 ms), excitation-команда **39,18%**. Допустимый диапазон обоих [0,1].
На 1 kg pass=9/11 и RMSE=3,572 mm. На всех пяти других массах pass=0/11,
без физического выхода из workspace; средние RMSE последней секунды
**20,48 / 23,79 / 49,74 / 93,73 / 151,57 mm** соответственно.
На hold 0 смещение +30,62 mm при 0,5 kg и −33,63 mm при 1,5 kg.
Большие peaks у тяжёлых масс включают заранее заданную reset-активацию,
а не освоенную компенсацию policy. Сохранённые веса не насыщают верхнюю
активацию в этих сценариях, но не удерживают требуемую высоту при новой массе.

Полный отчёт и ограничения: [vertical-mass-ppo.md](vertical-mass-ppo.md).
JSON: `runs/vertical_precision/mass_audit.json`; checkpoint SHA сохранён.

## Перелёт и торможение вертикального PPO — 2026-11-07

Команда:

```bash
taskset -c 0 env PYTHONPATH=/tmp/myo-supervised-plot-deps \
  MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/audit_vertical_precision_motion.py
```

Exit 0. Read-only повтор четырёх движений для старого/new best: восемь
траекторий по 4 s, физический шаг 1 ms. Инструментация оборачивает исходный
`mujoco.mj_step`, не меняя его вызов или физику; все 1600 границ control-шагов
совпали с сохранёнными t/q/v побитово. Обучение/selection не выполнялись.
Результаты: `runs/vertical_precision/combined/motion_audit.json`, PNG/SVG
`motion_comparison`; PNG просмотрен. `py_compile` скрипта и `git diff --check` прошли.

У старого best два перелёта вниз: **6,02/9,59 mm**, у нового все четыре
движения без пересечения цели на сетке 1 ms. Но установление подъёмов
замедлилось **0,445→1,001 s** и **0,409→0,695 s**. У новой модели при движении
вниз есть ранние небольшие реверсы до цели, поэтому отсутствие перелёта
не равно плавному монотонному торможению. Метрики и графики — в
[vertical-precision-ppo.md](vertical-precision-ppo.md#движение-перелёт-и-торможение).
Здесь установление означает пребывание в допусках ≤5 mm / ≤0,03 m/s на всех
оставшихся подшагах до конца 4 s; на более длительное время вывод не переносится.

## Более плотные задания и логарифмическая награда — 2026-11-07

Новый опыт только 4×128 SiLU RMSprop, variant 4. Вместо 9 — 33 train-сценария;
вместо экспоненты — log1p(abs(error)/5 mm), согласованный со старой кривой
при 0/80 mm. Остальные слагаемые, multiplier=0,1 и физика сохранены.
Полный протокол: [vertical-precision-ppo.md](vertical-precision-ppo.md).

Фактически выполнены:

```bash
.venv/bin/python3 -m pytest -q tests/test_vertical_precision.py tests/test_vertical_architecture.py
taskset -c 0,1 .venv/bin/python3 -m pytest -q
taskset -c 2 .venv/bin/python3 -m myo.train_vertical_architecture \
  --trial combined --protocol precision --output-root runs/vertical_precision --defer-test
```

Целевые проверки: **8 passed, 2,83 s**. Полный pytest внутри sandbox:
150 passed и один PermissionError в существующем loopback UDP-тесте.
Повтор полной команды вне sandbox: **151 passed, 6 прежних warnings, 42,58 s**.
Проверено точное совпадение физических переходов и прочих reward-компонентов,
цикл 33 заданий, сохранение/загрузка нужной архитектуры и rejection неверной
reward identity. Начальные веса совпадают с предыдущим combined побитово.
Для ошибки 5 mm потеря position reward выросла с 0,003314 до 0,131451.

Обучение завершено: **131 072 transitions, 4063 optimizer steps, 295,7 s**
с validation. Best выбран на 114 688 шагах: validation **1,861 mm, 9/9**;
last **3,278 mm, 7/9**. Дальнейшие фактические команды:

```bash
taskset -c 2 .venv/bin/python3 scripts/evaluate_vertical_precision.py
taskset -c 1 .venv/bin/python3 -m myo.vertical_precision \
  --controller runs/vertical_precision/combined/best.zip --seconds 12
taskset -c 0 env PYTHONPATH=/tmp/myo-supervised-plot-deps \
  MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/report_vertical_precision.py
```

Все exit 0. Test-сравнение с прежним combined/best: старые пять сценариев
**2,518→3,079 mm, 4/5→5/5**; расширенные одиннадцать
**2,868→3,572 mm, 9/11→9/11**. У новой модели не проходят удержания ±70 mm;
физических срывов у обеих нет. Смещение −30 mm остаётся **−4,952 mm**,
почти на пороге прохода. Убедительного улучшения точности нет.

Оценка подтвердила четыре строгие загрузки checkpoint, побитовое совпадение
начальных весов со старым опытом, воспроизводимый выбор по validation и
**11 точных native-повторов** best test. Headless demo: **12 s**, без срыва,
whole-run RMSE **14,342 mm** со сменами целей. График сохранён и просмотрен;
GUI не проверялся. Веса, JSON и график находятся в игнорируемом
`runs/vertical_precision/combined/`. `py_compile` новых модулей/скриптов и
`git diff --check` прошли. Новых обучений после просмотра test не запускали.

## Диагностика смещения равновесия — 2026-11-07

Read-only проверка сохранённого Muon, без переобучения и изменений физики.
Inline Python через `.venv/bin/python3` с affinity `taskset -c 0`:
`VerticalMuscles.reset(q)` и 200 native steps с постоянным исходным excitation
для q=−60/−30/0/+30/+60 mm; затем `load_policy` и по 200 deterministic
`model.predict`/`VerticalControlEnv.step` для девяти hold-целей от −60 до +60 mm
с шагом 15 mm. Для цели −30 mm отдельно проверены старты −40 и −20 mm.
Новые исходные состояния задавались только через штатный reset; ни один
переход или сила не корректировались. Результаты:
`runs/vertical_architecture/equilibrium_diagnostic.json`.

Постоянная сбалансированная команда удержала все пять высот; максимальный
дрейф менее 1e−8 mm за 4 s, численная погрешность. Muon при −30/0/+30 mm
даёт равновесия **−35,211 / −0,188 / +34,337 mm**. Два дополнительных старта
для цели −30 mm сходятся к **−35,211 mm**, конечные скорости порядка 1e−10 m/s.
При смещённом равновесии верхняя/нижняя силы **20,215 / 10,405 N**,
разность **9,81 N** равна mg. Это установившееся смещение, не недостаток силы
или продолжающийся переход. На промежуточных hold-целях ошибки больше, чем
на train-hold −60/0/+60 mm. Проверка не устанавливает причинную роль редкой
сетки целей: это гипотеза для отдельной абляции. Validation-hold только на 0
не проверяет постоянное удержание других высот. Потеря сырой position reward
при ошибке 5 mm **0,003314**, но это не доказательство, что effort делает
такое смещение оптимальным. Новые обучения не запускались.

Повтор отчётного скрипта exit 0; таблица равновесий добавлена в
[vertical-architecture-ppo.md](vertical-architecture-ppo.md).

## Архитектура и optimizers собственного PPO — 2026-11-07

Опыт начат 2026-11-06, завершён 2026-11-07. Отдельные таблицы для активации,
глубины, optimizer и их комбинации, кривые и ограничения:
[vertical-architecture-ppo.md](vertical-architecture-ppo.md).
Это нативный PPO-позиционер на механике 4, не MuscleMimic imitation:
50 Hz control, 1 kHz физика, независимое упругое сухожилие, FIFO 40 ms,
activation dynamics 15/50 ms. Награда, входы и Beta-команды не менялись.

Фактические команды обучения (первые шесть выполнялись параллельно):

```bash
taskset -c 0 .venv/bin/python3 -m pytest -q tests/test_vertical_architecture.py
taskset -c 2 .venv/bin/python3 -m myo.train_vertical_architecture --trial control --steps 131072 --eval-every 8192 --defer-test
taskset -c 3 .venv/bin/python3 -m myo.train_vertical_architecture --trial silu --steps 131072 --eval-every 8192 --defer-test
taskset -c 4 .venv/bin/python3 -m myo.train_vertical_architecture --trial deep --steps 131072 --eval-every 8192 --defer-test
taskset -c 5 .venv/bin/python3 -m myo.train_vertical_architecture --trial radam --steps 131072 --eval-every 8192 --defer-test
taskset -c 6 .venv/bin/python3 -m myo.train_vertical_architecture --trial rmsprop --steps 131072 --eval-every 8192 --defer-test
taskset -c 7 .venv/bin/python3 -m myo.train_vertical_architecture --trial muon --steps 131072 --eval-every 8192 --defer-test
taskset -c 2 .venv/bin/python3 scripts/train_vertical_combined.py
taskset -c 0,1 .venv/bin/python3 -m pytest -q
taskset -c 0 .venv/bin/python3 scripts/benchmark_vertical_native_optimizers.py
taskset -c 0 .venv/bin/python3 scripts/evaluate_vertical_architecture.py --include-combined
taskset -c 0 env PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_vertical_architecture.py
taskset -c 1 .venv/bin/python3 -m myo.vertical_control --controller runs/vertical_architecture/muon/best.zip --seconds 12 --targets .06 -.06 0 --hold-seconds 4
.venv/bin/python3 -m py_compile src/myo/optimizer_muon.py src/myo/train_vertical_architecture.py scripts/train_vertical_combined.py scripts/evaluate_vertical_architecture.py scripts/report_vertical_architecture.py scripts/benchmark_vertical_native_optimizers.py tests/test_vertical_architecture.py
git diff --check
```

Все команды exit 0. Targeted pytest: **6 passed**, **2,50 s**.
Полный pytest с разрешённым существующим loopback UDP: **149 passed**,
шесть прежних предупреждений, **60,59 s**. В этой серии нового sandbox-failure
нет: полный набор сразу запущен вне sandbox по ранее установленному ограничению
локального UDP. Для всех шести конфигураций проверены PPO update, перенос
архитектуры и optimizer moments, а также совпадение следующего обновления после
загрузки при одинаковых градиентах/LR. Последнее защищает смешанный Muon/Adam
от потери связи внутренних optimizer с восстановленными группами и состояниями.

Шесть однофакторных опытов и одна комбинация: **131 072 transitions на опыт**,
**917 504 всего**, seed=0, с нуля. Сравниваются 2×128 Tanh Adam; SiLU;
4×128 Tanh; RAdam; RMSprop; PyTorch Muon+Adam fallback. Одинаковые исходные
weights всех пяти неглубоких сетей подтверждены SHA256. У deep и combined
тоже одинаковые исходные weights, но SiLU меняет вычисление активаций.
Контроль содержит **37 253**, глубокие сети — **103 301** параметр.
Общие PPO параметры/версии/physical contract совпадают; LayerNorm не добавлен.

Для каждого опыта выполнено ровно 128 rollout по 1024 transitions, без остановки
по плато. Validation каждые 8192. Из-за target KL=0,03 реальное число minibatch
optimizer updates различается; оно сохранено отдельно, не подменяется числом
эпох SB3. LR=0,0003 общий, отдельного подбора под optimizer здесь не было.
Комбинация 4×128 SiLU RMSprop выбрана только после завершения независимых
SiLU/deep/RMSprop и их выигрыша по validation над контролем. Основания выбора
сохранены до обучения в `combination_protocol.json`. Все семь выборов best
зафиксированы до первых test-прогонов в `frozen_selection.json`.

| Опыт | Best validation steady, mm; pass | Test best steady, mm; pass | Test full, mm | Test last steady, mm; pass |
| --- | --- | --- | ---: | --- |
| control | 3.71; 4/5 | 3.99; 4/5 | 8.50 | 3.99; 4/5 |
| silu | 2.10; 5/5 | 4.27; 3/5 | 8.58 | 8.82; 1/5 |
| deep | 3.13; 5/5 | 2.92; 4/5 | 7.54 | 2.65; 4/5 |
| radam | 10.88; 1/5 | 8.61; 2/5 | 12.51 | 8.61; 2/5 |
| rmsprop | 2.45; 5/5 | 5.18; 4/5 | 10.05 | 7.24; 2/5 |
| muon | 1.81; 5/5 | 2.44; 4/5 | 7.76 | 2.44; 4/5 |
| combined | 1.94; 5/5 | 2.52; 4/5 | 7.95 | 5.06; 3/5 |

Steady — средняя RMSE последней секунды по пяти сценариям; full — вся
четырёхсекундная траектория. Приёмка: |position error|≤5 mm и |velocity|≤0,03 m/s
не менее 90% последней секунды, без physical failure. У всех best test-моделей
в этом опыте физические срывы отсутствовали. Полные траектории и покейсные
результаты сохранены; средние не подменяют строгую приёмку. Ни одна модель
не прошла 5/5 test. Единственная проблема Muon/deep/combined — hold −30 mm
со смещением **5,21 / 5,39 / 5,32 mm** соответственно. У SiLU/RMSprop меньше
ошибка удержаний, но хуже перенос на новые цели движения. Вмешательства
в состояние физики или учительских возбуждений не было.

Muon best совпал с последним checkpoint: фиксированный бюджет не доказывает
плато. SiLU/RMSprop/combined ухудшились после лучших checkpoints. Увеличение
глубины одновременно увеличивает ёмкость; опыт не отделяет этот эффект от
глубины при одинаковом числе параметров. Все результаты — один seed и один LR.

Независимый аудит: **21** строгая загрузка initial/best/last, **35** точных native
replay test-траекторий по записанным возбуждениям, повтор validation и проверка
best selection. Проверены равенство native q/v/time, physical/action/source
contracts и optimizer step counters. Test выполнен для best и last после
окончания отбора. Отчёт: `runs/vertical_architecture/independent_evaluation.json`.

Отдельный kernel benchmark: 10 прогревочных steps и 30 чередующихся замеров
на одном CPU thread, одинаковые начальные параметры и синтетические gradients;
без forward/backprop, clipping и физики. Median optimizer.step:
Adam **0,799 ms**, RAdam **1,251 ms**, RMSprop **0,563 ms**, Muon **71,693 ms**.
Muon использует PyTorch bfloat16 NS; это не проверенная численная копия Optax.
Дополнительный inline Python с `torch.utils.benchmark.Timer(...).blocked_autorange`
на том же CPU измерил `128×128 @ transpose`: float32 **0,121 ms**, bfloat16
**1,008 ms**, отношение **8,31**. PyTorch сообщает capability AVX2. Этот замер
включён также в скрипт для воспроизведения. Это вклад dtype/backend, не доказанное
ускорение полного Muon при замене NS на float32: такой controller не обучался.
Wall time обучения не является чистым kernel benchmark: опыты работали
параллельно и имели разное число optimizer steps из-за target KL.

Headless demo Muon через публичный CLI закончил **12 s** без physical failure;
общая RMSE **14,25 mm** включает переходы между +60/−60/0 mm и не равна
steady RMSE. Результат сохранён в `headless_demo.json`. Новое GUI не проверялось.
PNG/SVG экспортированы и просмотрены. Веса и rollout остаются в ignored
`runs/vertical_architecture`; production defaults и прежние checkpoints не менялись.

## Adam против Muon на MuscleMimic — 2026-11-06

Протокол, таблицы, test checkpoint comparison и графики:
[vertical-mimic-optimizers.md](vertical-mimic-optimizers.md).
Меняется только optimizer на механике 4: Adam (upstream AdamW с decay=0,
eps=1e−5) вместо Muon. Сохраняются LR=0,0004, сеть 4×256, seed=0, reward,
нормализация, GAE, native-физика, 128 сред, 20 шагов, 32 minibatch и одна эпоха.
128 PPO-итераций = 327 680 transitions = 4096 optimizer steps.
Побитово проверены одинаковые исходные params/run_stats; initial validation,
общие training settings, physical contract и версии библиотек совпадают.

Фактические команды:

```bash
taskset -c 0,1 .venv/bin/python3 -m pytest -q tests/test_vertical_mimic.py
.venv/bin/python3 -m myo.train_vertical_mimic --variant 4 --optimizer adam --output-dir runs/vertical_mimic_adam/variant_4 --num-envs 128 --updates 128 --hidden 256 256 256 256 --eval-every 16 --seed 0 --cpu-threads 2
taskset -c 4,5 .venv/bin/python3 -m pytest -q
taskset -c 4,5 .venv/bin/python3 scripts/evaluate_vertical_mimic.py --root runs/vertical_mimic_adam --variants 4 --checkpoint-progress
.venv/bin/python3 scripts/benchmark_vertical_mimic_optimizers.py
taskset -c 0,1 env PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_vertical_mimic_optimizers.py
.venv/bin/python3 -m py_compile src/myo/train_vertical_mimic.py scripts/benchmark_vertical_mimic_optimizers.py scripts/report_vertical_mimic_optimizers.py tests/test_vertical_mimic.py
git diff --check
```

Targeted pytest: **5 passed**, 11,11 s. Полный pytest: **143 passed**,
шесть прежних warnings, 56,14 s. Первый полный запуск внутри sandbox:
142 passed и один failure из-за запрета существующего loopback UDP-сокета;
повтор вне sandbox прошёл. Проверяется перенос обоих разных optimizer states,
а для Muon — совместимость с точно закреплённым прежним trainer hash.
Неизвестные hash и несовместимые physical/source/weight contracts не разрешаются.

Обучение Adam завершилось за **393.74 s**. Лучший validation
checkpoint — iteration 48, q RMSE **30,91 mm**; last 128 — **48,67 mm**.
Test best: **29,78 mm**, **0,0639 m/s**, completion **5/5**, strict **0/5**.
Для прежнего Muon при том же бюджете: best/last validation **18,16 mm**,
test **20,62 mm**, **0,0522 m/s**, completion **5/5**, strict **0/5**.
Adam точнее в отложенном статическом hold (13,92 против 22,80 mm),
Muon — на четырёх moving clips. Один seed и один LR не доказывают
превосходства optimizer в общем случае.

По вопросу пользователя после обучения дополнительно оценены initial и last
на test: **94,24 → 53,79 mm**; validation-best 48 даёт **29,78 mm**.
Таким образом, есть обучение относительно старта и деградация после лучшего
checkpoint. Это post-hoc диагностика трёх checkpoints, не test-based selection.
Команда исполняла `load_checkpoint` и `evaluate_mimic(..., TEST_MOTIONS)`
для initial/last; затем тот же путь повторён через `--checkpoint-progress`.
Отчёты `initial_test.json`, `last_test.json`, `test_progress.json`
сохранены рядом с основным `test.json`.

Независимый аудит: **3** строгих загрузки,
**5** точных native replay; test-метрики повторились.
Проверены best selection по validation, direct actions, delivered excitation,
activation и native q/v без подмены механики.

Одинаковый готовый batch (2560 transitions), одна архитектура/инициализация,
шесть чередующихся замеров каждого optimizer после отдельной JAX-компиляции:
медиана полного PPO-update Muon **2.2897 s**,
Adam **0.1961 s**, отношение **11.68**.
Из замера исключены сбор опыта и validation; это полный loss/backprop/update,
не отдельный optimizer kernel. Параллельный native-аудит работал на других
CPU cores; частоты/тепловой режим не фиксировались. Отношение wall time старого
и нового обучения не является чистой оценкой optimizer: старые четыре варианта
обучались одновременно. Все сырые времена сохранены в `update_benchmark.json`.

Веса/траектории/PNG/SVG остаются в ignored `runs/vertical_mimic_adam`.
Windows/GPU и новое GUI не проверялись; изменение касается headless PPO-опыта.

## Перенос MuscleMimic на вертикальный стенд — 2026-11-06

Протокол, различия с исходной конфигурацией, таблицы и графики —
[vertical-musclemimic.md](vertical-musclemimic.md).
Исполняются оригинальные JAX `networks.py`, `optimizer.py`, `loss.py` и
`rollout_buffer.py` из commit `47b653db2bce25de834360d2ee6b2dd88142bd78`.
Публичные исходники и LICENSE подготовлены в игнорируемом
`data/musclemimic_methodology`, checksums закреплены в
`configs/vertical_mimic_sources.json`. Чужих weights, моделей и motion capture нет.

Фактические команды (четыре variant-команды выполнялись параллельно):

```bash
uv pip install --python .venv/bin/python3 'jax==0.7.2' 'flax==0.12.0' 'optax==0.2.6' 'distrax==0.1.7'
taskset -c 0,1 .venv/bin/python3 -m pytest -q
taskset -c 2,3 .venv/bin/python3 -u -m myo.train_vertical_mimic --variant 1 --output-dir runs/vertical_mimic/variant_1 --num-envs 128 --updates 128 --hidden 256 256 256 256 --eval-every 16 --cpu-threads 2
taskset -c 4,5 .venv/bin/python3 -u -m myo.train_vertical_mimic --variant 2 --output-dir runs/vertical_mimic/variant_2 --num-envs 128 --updates 128 --hidden 256 256 256 256 --eval-every 16 --cpu-threads 2
taskset -c 6,7 .venv/bin/python3 -u -m myo.train_vertical_mimic --variant 3 --output-dir runs/vertical_mimic/variant_3 --num-envs 128 --updates 128 --hidden 256 256 256 256 --eval-every 16 --cpu-threads 2
taskset -c 8,9 .venv/bin/python3 -u -m myo.train_vertical_mimic --variant 4 --output-dir runs/vertical_mimic/variant_4 --num-envs 128 --updates 128 --hidden 256 256 256 256 --eval-every 16 --cpu-threads 2
taskset -c 0,1 .venv/bin/python3 -u scripts/evaluate_vertical_mimic.py --variants 1 --output runs/vertical_mimic/independent_1.json
taskset -c 2,3 .venv/bin/python3 -u scripts/evaluate_vertical_mimic.py --variants 2 --output runs/vertical_mimic/independent_2.json
taskset -c 4,5 .venv/bin/python3 -u scripts/evaluate_vertical_mimic.py --variants 3 --output runs/vertical_mimic/independent_3.json
taskset -c 0,1 .venv/bin/python3 -u scripts/evaluate_vertical_mimic.py --variants 4 --output runs/vertical_mimic/independent_4.json
taskset -c 0,1 .venv/bin/python3 scripts/demo_vertical_mimic.py --controller runs/vertical_mimic/variant_1/best.msgpack
taskset -c 2,3 .venv/bin/python3 scripts/demo_vertical_mimic.py --controller runs/vertical_mimic/variant_4/best.msgpack
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_vertical_mimic.py
git diff --check
```

Все финальные команды exit 0. Полный pytest вне sandbox для существующего
loopback UDP-теста: **142 passed**, шесть предупреждений Gymnasium/SB3,
**95,13 s**. В первом sandbox-прогоне UDP-сокет был запрещён: 140 passed и один
такой failure до добавления reward-parity регрессии. Вне sandbox набор прошёл.
Проверены reference-reset без teacher activation, preview и конец клипа,
прямой Gaussian→excitation mapping, native transport 40 ms, границы GAE,
PPO-update, shape двух actions, сериализация и строгая загрузка checkpoint.
Отдельная регрессия исполняет арифметику исходного `MimicReward` через AST:
покомпонентное xyz-среднее, включая нулевые координаты, совпадает с адаптером.

Основной опыт: **327 680 transitions на вариант**, суммарно **1 310 720**,
seed=0, CPU, четыре скрытых слоя 256 вместо 1024 у закреплённого source-config.
Полный source-размер доступен через CLI; замер с 32 средами дал 136,65/140,90 s
на первые два updates у 4×1024 против 2,93/2,74 s после компиляции у 4×256.
Опубликованный 16-слойный checkpoint, Warp, 8192 среды и миллиардный бюджет
не воспроизведены. Четыре основные policy обучены с нуля; чужих возбуждений,
predictor и градиента через физику нет. Частота controller 100 Hz, physics 1 kHz.

Итоговые test mean position RMSE: **11,58 / 10,75 / 18,97 / 20,62 mm**.
Все четыре модели закончили **5/5** клипов по исходному широкому порогу 25 cm,
но строгую приёмку (полный клип, q RMSE ≤10 mm и v RMSE ≤0,05 m/s)
прошли только **2/5, 2/5, 1/5, 0/5**. Новое статическое удержание +20 mm
не прошло ни у одной. При stochastic actions строгих успехов 0/5 у всех;
демо использует средние выходы policy. Это частично выученное слежение,
не подтверждение произвольного управления ускорением или готового D1.
Выполнимость высокой точности на механике 1 отдельно показывает диагностический
reference-контроллер: **5/5**, mean q RMSE **0,40 mm**. Он не обучает PPO.

Четыре независимых отчёта сведены в `runs/vertical_mimic/independent_evaluation.json`
с hashes частей: **12** строгих загрузок initial/best/last и **20** точных native
replay по записанным excitations, включая delivered/activation, q/v и время.
Повторены test-метрики и проверен отбор best исключительно по validation.
Main-trial summaries/versions/source snapshots — в `runs/vertical_mimic`;
checkpoints и данные остаются ignored. Прямого количественного сравнения с
прежней acceleration-задачей как однофакторной абляцией нет.

Предварительные сбои scalar shape и NumPy JSON-типов исправлены и покрыты
регрессиями. Отдельный pilot с квадратом нормы вместо xyz mean остановлен
до test: 46/27/26/13 полных updates, **286 720** завершённых transitions,
архив `runs/vertical_mimic_site_mean_pilot`; в итоговые веса он не переносился.
Его стоимость, начальные smoke/benchmarks не входят в основной бюджет выше.

Headless demo: оба закончились без физического срыва за **6,20 s**;
q RMSE первого test-клипа **9,15 mm** у варианта 1 и **20,43 mm** у варианта 4.
PNG/SVG сравнений сохранены в `runs/vertical_mimic`. Новое GUI и Windows/GPU
в этом опыте не проверены. Протокол работает локально без cloud tracking.

## Задания ускорения через 40 ms — 2026-11-06

```bash
.venv/bin/python3 -m pytest -q tests/test_vertical_scheduled.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 scripts/run_vertical_scheduled.py > runs/vertical_scheduled_ppo.log 2>&1
.venv/bin/python3 scripts/evaluate_vertical_scheduled.py > runs/vertical_scheduled_ppo_evaluation.log 2>&1
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_vertical_scheduled.py
.venv/bin/python3 scripts/demo_vertical_scheduled.py --controller runs/vertical_scheduled_ppo/variant_4/best.zip > runs/vertical_scheduled_ppo_demo.log 2>&1
.venv/bin/python3 -m py_compile src/myo/vertical_scheduled.py src/myo/train_vertical_scheduled.py scripts/run_vertical_scheduled.py scripts/evaluate_vertical_scheduled.py scripts/report_vertical_scheduled.py scripts/demo_vertical_scheduled.py tests/test_vertical_scheduled.py
git diff --check
git check-ignore runs/vertical_scheduled_ppo/summary.json runs/vertical_scheduled_ppo/variant_4/best.zip
```

Финальные команды exit 0. Первые три проверки — 3 passed, 4,98 s;
после регрессии отбора — **4 passed**, **3,85 s**. Финальный полный набор
вне sandbox для существующего loopback UDP-теста — **138 passed**, шесть
предупреждений Gymnasium/SB3, **40,71 s**. Проверены отдельная issued/due FIFO,
оценка задания на правильном интервале, отсутствие невыданного будущего во
входе, native-моторная задержка/активация, опустошение очереди перед truncation,
неуспех goal-blind удержания, PPO-update и несовместимость со старым task.

Сеть теперь получает прямые ускорения, без цели положения и PD-эталона:
команда из t относится к среднему ускорению t+40…t+60 ms, reward вычисляется
в t+60 ms. Во входе три уже выданных задания на текущий интервал и через
20/40 ms; итого **7/9/13/15** признаков. Цель не вычисляется из будущего
симулированного состояния. Reward: Gaussian ошибки среднего ускорения с
масштабом 0,5 m/s², исходные малые effort/change и failure; бонуса за
нулевую скорость нет. Физика, груз 1 kg и активации прежние.

Семь фиксированных train-профилей, пять validation и пять test, без перебора
сидов. Знакочередующиеся команды A=0,5…1 m/s² с базовой лобой 100–200 ms,
два сбалансированных цикла и два шага для завершения последних заданий.
Профиль успешен при RMSE≤0,25 m/s² и ≥80% интервалов в допуске 0,35 m/s²,
без физического падения. Все переходы команд входят в метрику. Максимальный
бюджет 131072 на механику, остановка также по плато/повторной готовности.

В первоначальном запуске выявлен дефект отбора: один нулевой успех с RMSE
0,414 скрывал улучшение движений до RMSE 0,399 без порогового успеха.
Отбор/остановка исправлены: failures, полная готовность, mean RMSE,
passed только при равенстве. Критерий изменён до test обученных PPO. Первый запуск
прерван SIGINT, exit 130; **376832** выполненных шагов и точные исходники
сохранены в `runs/vertical_scheduled_threshold_selection/`, PPO test не вычислялся.
Ветки 1/4 повторены. У веток 2/3 passed всегда оставался нулём, поэтому
порядок и момент остановки не изменились; их **196608** шагов переиспользованы
с исходными metadata и hash trainer из архива, без переписывания происхождения.

Итоговые четыре trial содержат **401408** переходов среды, из них **204800**
выполнены в финальном запуске. С учётом первого запуска — **581632** реально
выполненных обучающих перехода. Wall time финального запуска **460,97 s**;
CPU, до четырёх процессов по одному torch-потоку, seed 0. Все четыре остановлены
по плато. Best: **32768/73728/24576/81920**, всего шагов в trial:
**73728/122880/73728/131072**. Не менялись сеть 128×128 Tanh, Beta-excitations
и исходные hyperparameters PPO; предиктора и teacher actions нет.

Test PPO: **0/5, 0/5, 0/5, 1/5**, mean acceleration RMSE
**0,671 / 0,697 / 1,027 / 0,436 m/s²**, физических падений нет.
Успешных ненулевых test-профилей **0/16**; прошёл только нулевой профиль полной
механики. При этом её коэффициент cov(actual,due)/var(due) **0,674** —
частичная реакция на команды есть, но точное исполнение не освоено.
Постоянное удержание даёт **1/5** во всех механиках, ошибки
**0,537/0,541/0,551/0,548 m/s²**. Простая силовая инверсия:
**5/5, 5/5, 1/5, 1/5**, **0,083/0,123/0,322/0,379 m/s²**.
Это диагностические baseline, не обучающие метки; инверсия не моделирует
точно будущие активацию/compliance и не устанавливает оптимум механик 3/4.

Силовые baseline на test-сценариях предварительно проверялись для оценки
выполнимости. `summary.json`: status `complete`, выбор всех четырёх best
зафиксирован до вычисления test обученных PPO.
`independent_evaluation.json`: status `passed`, **12** строгих initial/best/last
загрузок, повторены отбор, initial/last validation, train/validation/test,
native replay и исходные источники переиспользованных веток. **20** test-траекторий
проверены независимо по выданным/действующим заданиям, Δv/Δt, длительности
и точному совпадению native q/v/time. Общие префиксы повторного/первого запуска
совпали точно по physical validation/PPO logs/rollout audits:
**65536/122880/73728/114688** шагов. В rollout audit нет команд вне [0,1].

Headless demo полной механики прошло весь test-профиль **1,44 s**, без падения,
RMSE **0,412 m/s²**, без автоматического повторного reset. Новый GUI не запускался.
Два графика PNG/SVG экспортированы и просмотрены. Протокол и таблицы — в
[vertical-scheduled-ppo.md](vertical-scheduled-ppo.md). Старые физика, predictor
и checkpoints не менялись; новая задача не является чистой абляцией одного
тайминга относительно прежнего PD-опыта. Один seed ограничивает выводы.
Сгенерированные веса и отчёты остаются ignored; внешних сервисов/новых assets нет.

## PPO с наградой за ускорения — 2026-11-06

```bash
.venv/bin/python3 -m pytest -q tests/test_vertical_acceleration.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo.train_vertical_acceleration --variant 4 --objective position --max-steps 8192 --min-steps 8192 --output-dir runs/vertical_acceleration_control_preflight > runs/vertical_acceleration_control_preflight.log 2>&1
.venv/bin/python3 scripts/run_vertical_acceleration.py > runs/vertical_acceleration_ppo.log 2>&1
.venv/bin/python3 scripts/evaluate_vertical_acceleration.py > runs/vertical_acceleration_ppo_evaluation.log 2>&1
.venv/bin/python3 scripts/audit_vertical_acceleration_reward.py > runs/vertical_acceleration_reward_audit.log 2>&1
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_vertical_acceleration.py
.venv/bin/python3 scripts/demo_vertical_acceleration.py --controller runs/vertical_acceleration_ppo/variant_4/best.zip --seconds 12 > runs/vertical_acceleration_ppo_demo.log 2>&1
.venv/bin/python3 -m py_compile src/myo/vertical_acceleration.py src/myo/train_vertical_acceleration.py scripts/run_vertical_acceleration.py scripts/evaluate_vertical_acceleration.py scripts/audit_vertical_acceleration_reward.py scripts/report_vertical_acceleration.py scripts/demo_vertical_acceleration.py tests/test_vertical_acceleration.py
git diff --check
git check-ignore runs/vertical_acceleration_ppo/summary.json runs/vertical_acceleration_ppo/variant_4/best.zip
```

Команды exit 0. Новые регрессии — **3 passed**, **1,90 s**. Полный набор
вне sandbox (существующий loopback UDP-тест) — **134 passed**, четыре прежних
предупреждения Gymnasium/SB3, **34,23 s**. Проверены знак и среднее ускорение,
PD-эталон из состояния до перехода, неизменный native-переход, сохранение
транспортной задержки и постепенной активации, PPO-update и отказ переносить
checkpoint в среду с позиционной наградой. Новая награда заменяет только
позиционное слагаемое; скорость, effort, изменение команды и failure сохранены.

Эталон `a_ref = 28*(target-q_before)-9*v_before`, реальное ускорение
`(v_after-v_before)/elapsed_time`, масштаб 2,24 m/s². PD-часть heuristic без
preview служит инженерным заданием ускорения. Сеть получает прежние состояние
и цель положения. Модели динамики, градиента через физику и regression loss нет:
actor/critic обучаются настоящим PPO/GAE. Обновление reward не делает действие
мгновенным; первые 40 ms у механик 3/4 по-прежнему независимы от новой команды.

Четыре policy с нуля, seed 0, actor/critic 128×128 Tanh и исходные hyperparameters.
До обучения тензоры каждой точно совпали с исходным initial checkpoint.
Дополнительный контрольный запуск **8192 шагов** с позиционной наградой точно
повторил первый блок исходной механики 4: физические метрики, PPO losses и
восемь rollout audits. Контрольные результаты взяты из `runs/vertical_ppo/`,
его hash закреплён. Результаты дообучения из sweep сюда не подмешаны.

Основная серия — **475136 переходов среды по 20 ms**, **475,68 s** wall time,
четыре локальных процесса по одному torch-потоку. Механики 1/2/3 дошли до
предела **131072**; их плато не установлено. Полная механика 4 остановилась
после **81920** по шести проверкам без существенного улучшения. Best на шагах
**131072 / 131072 / 122880 / 32768**, все выбраны по position-validation
до открытия test, не по ошибке ускорения.

Test: по **1/5** успехов у всех четырёх; mean last-second RMSE
**9,315 / 19,485 / 22,891 / 16,467 mm**. Исходный position-PPO:
**5/5, 5/5, 2/5, 3/5**, **0,359 / 2,887 / 6,256 / 4,213 mm**.
Новая схема хуже в этом запуске; это не утверждение о невозможности PPO
обучаться по ускорению вообще. Выбранные модели прошли test без физических
падений, но точное управление не освоили. Непрерывный headless-загрузчик полной
механики прошёл 12 s с +80/−80/0 mm без reset/падения; полный RMSE с переходами
**34,794 mm**. GUI этого нового загрузчика не запускался.

`summary.json`: status `complete`. `independent_evaluation.json`: status
`passed`, **20** строгих загрузок новых/контрольных checkpoint. Повторены
initial/best/last, initial weights, выбор best, last validation,
validation/test/switches обеих схем, источник и контрольный префикс.
В rollout audit нет команд за [0,1].

`reward_audit.json`: status `passed`. Команды **40** детерминированных test-эпизодов
перепроиграны в native-среде с новой наградой, время/q/v совпали точно.
Выполнены ещё **40** test-эпизодов со стохастическими действиями, fixed seed 0,
без обучения и физических падений. Полный acceleration RMSE у новых моделей
меньше, но steady acceleration RMSE хуже. Средняя новая raw-награда при
детерминированных командах выше у исходных policy, при stochastic seed 0 —
у новых. Например, механика 4: deterministic **0,9787 → 0,9321**,
stochastic **0,6480 → 0,7379**. Последние обучающие распределения заметно
сузились, средняя нижняя команда приблизилась к нулю. Это основание проверить
влияние шума на поиск, а не доказательство единственной причины неудачи.
Один seed и разные посещаемые состояния ограничивают вывод.

Два графика PNG/SVG экспортированы и просмотрены. Формулы, таблицы и разбор —
в [vertical-acceleration-ppo.md](vertical-acceleration-ppo.md). Источники прежней
физики, predictor и policy не менялись; веса/журналы остаются ignored.

## Однофакторный подбор PPO на вертикальном стенде — 2026-11-06

```bash
.venv/bin/python3 -m pytest -q tests/test_vertical_sweep.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 scripts/run_vertical_sweep.py > runs/vertical_ppo_sweep.log 2>&1
.venv/bin/python3 scripts/evaluate_vertical_sweep.py > runs/vertical_ppo_sweep_evaluation.log 2>&1
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/report_vertical_sweep.py --output docs/vertical-ppo-sweep.md
.venv/bin/python3 scripts/demo_vertical_sweep.py --controller runs/vertical_ppo_sweep/variant_4/control/best.zip --seconds 12 > runs/vertical_ppo_sweep_demo.log 2>&1
.venv/bin/python3 -m py_compile src/myo/vertical_sweep_env.py src/myo/train_vertical_sweep.py scripts/run_vertical_sweep.py scripts/evaluate_vertical_sweep.py scripts/report_vertical_sweep.py scripts/demo_vertical_sweep.py tests/test_vertical_sweep.py
git diff --check
git check-ignore runs/vertical_ppo_sweep/summary.json runs/vertical_ppo_sweep/variant_4/control/best.zip
```

Команды exit 0. Три новые регрессии — **3 passed**, 2,03 s; полный набор
вне sandbox для существующего loopback UDP-теста — **131 passed**, четыре
предупреждения Gymnasium/SB3, **31,61 s**. Проверяются независимость reward
от физического перехода, отсутствие мутации общего reward, сохранение весов
и моментов Adam при смене параметров, действительная смена buffer gamma/lambda,
размера rollout и schedules, фактические updates и строгий перенос checkpoint.
В матрице каждое изменение относится ровно к одному фактору.

Завершены **48 ветвей**: 16 режимов, включая продолженный контроль, × механики
2/3/4. По **32768 новых переходов среды по 20 ms**; всего **1 572 864**,
**109 198** шагов Adam, wall time оркестратора **2066,48 s**. Факторы проходят
последовательно; три независимые механики выполняются до трёх локальных
процессов, каждый с одним torch-потоком. Исходная механика и predictor не
менялись. Во всех ветвях своей механики один best-родитель, его Adam и seed 0,
свежий reset и одинаковое начало цикла девяти train-задач. Ранней остановки нет.
Это локальные продолжения, не сравнение с нуля и не комбинация улучшений.
При изменении gamma/GAE/reward critic адаптируется из исходного режима.

Лучшие по validation изменения — ширина позиционной награды 20 mm для механик
2/3 и clipping 0,1 для механики 4. Их test RMSE последней секунды —
**1,231 / 3,741 / 2,855 mm**, у продолженного контроля —
**2,075 / 4,535 / 4,473 mm**. Число успешных test-случаев этих ветвей и
контроля одинаково: **5/5, 3/5, 3/5**. Некоторые другие настройки полной
механики дают 4/5 test; ни одна не дала всех 5/5 validation и test.
У механик 3/4 оставшиеся непройденные test-цели выбранных ветвей — ±30 mm,
ошибки 5,36–5,92 mm при допуске 5 mm. Все выбранные checkpoint прошли
validation/test без физических падений. Best механик 2/3 найден на конце
бюджета, поэтому плато для них не установлено. Один seed и фиксированный
набор задач не устанавливают статистический оптимум гиперпараметров.

`runs/vertical_ppo_sweep/summary.json`: status `complete`. Выбор всех весов
зафиксирован **до первого test**. Независимый отчёт
`independent_evaluation.json`: status `passed`, строго загружены **96**
best/last checkpoint и три исходных родителя. Повторены validation для обоих
снимков, test/switches для best, выбор best из истории, параметры PPO/буфера,
счётчики переходов/Adam и hash весов. В сохранённом аудите нет команд за [0,1].
Для сравнения exploration STD сохранены одинаковые причинные probe-наблюдения
родительской policy. Контроль механик 2/3 точно повторил значения history/audit
последовательного preflight; архив `vertical_ppo_sweep_sequential_preflight`
в итоговые 48 ветвей не включён.

Новый загрузчик расширенного контракта прошёл headless 12 s полной механики
с целями +80/−80/0 mm без reset и падения; RMSE с переходами **17,971 mm**.
Это другая метрика, чем установившаяся ошибка таблиц. В этом опыте новый GUI
не запускался. Три графика PNG/SVG экспортированы и просмотрены; десять
таблиц, протокол, actual updates, best/last и ограничения — в
[vertical-ppo-sweep.md](vertical-ppo-sweep.md). Все веса/журналы остаются ignored;
исходные checkpoint сохранены. Новых assets, облачных запусков и сервисов нет.

## Прямой PPO на вертикальном стенде — 2026-11-06

```bash
.venv/bin/python3 -m pytest -q tests/test_vertical_control.py
.venv/bin/python3 -m myo.train_vertical_ppo > runs/vertical_ppo.log 2>&1
.venv/bin/python3 -m pytest -q
.venv/bin/python3 scripts/evaluate_vertical_ppo.py > runs/vertical_ppo_evaluation.log 2>&1
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/plot_vertical_ppo.py
.venv/bin/python3 -m myo.vertical_control --controller runs/vertical_ppo/variant_4/best.zip --seconds 12 > runs/vertical_ppo_demo.log 2>&1
.venv/bin/python3 -X faulthandler -m myo.vertical_control --controller runs/vertical_ppo/variant_4/best.zip --view --seconds 0.3 > runs/vertical_ppo_viewer.log 2>&1
.venv/bin/python3 -m py_compile src/myo/vertical_control.py src/myo/train_vertical_ppo.py scripts/evaluate_vertical_ppo.py scripts/plot_vertical_ppo.py tests/test_vertical_control.py
git diff --check
```

Рабочие команды exit 0. Три новых регрессии — **3 passed**, 4,39 s;
финальный полный набор вне sandbox (нужен существующему UDP-тесту) —
**128 passed**, четыре предупреждения Gymnasium/SB3, **36,17 s**.
Проверяются точное совпадение native-переходов с обёрткой, независимая
задержка/активация, смена цели без reset, отказ считать удержание без цели
или раннее падение успехом, PPO-update, сериализация отчёта и перенос
checkpoint с отказом при другой механике. Ошибки пустых reset options и
float32 в JSON были исправлены на подготовке до рабочего обучения.

Исходная физика вертикального стенда и предиктора не менялась. Четыре policy
с нуля: actor/critic 128×128 Tanh, прямые Beta-excitations, PPO по награде
за положение/скорость и малым затратам. Предиктора и размеченных действий
эвристики нет. Seed 0, локальный CPU, один torch-поток. Девять фиксированных
train-эпизодов по 4 s, пять validation и пять test, отдельная проверка
смены команд. Все параметры и reward сохранены в metadata и
[vertical-ppo.md](vertical-ppo.md).

Завершено **311 296** переходов за **706,69 s**, status `complete` в
`runs/vertical_ppo/summary.json`. Механики 1/2: 32768/65536 шагов,
остановка после повторных 5/5 validation. Механики 3/4: 98304/114688,
остановка по плато, лучшие validation 3/5 и 4/5.

У выбранных моделей test пройден **5/5, 5/5, 2/5, 3/5**, средний RMSE
последней секунды **0,359 / 2,887 / 6,256 / 4,213 mm**. Это средние
по случаям, не замена требования прохождения каждого случая. Падений у
выбранных моделей на validation/test нет. Известные train-случаи:
**9/9, 9/9, 8/9, 6/9**. Полное освоение delayed-вариантов не установлено.

Независимая оценка `independent_evaluation.json`, status `passed`:
загружены **12** initial/best/last checkpoint, проверены hash/контракты,
выбор весов по падениям/успехам/ошибке, повторены validation/test/switches.
В PPO buffers не было команд вне [0,1]. Все четыре непрерывных 12-секундных
прогона через учебный time limit дошли до конца без падения. При увеличении
постоянного удержания validation с 4 до 12 s результаты 5/5, 5/5, 3/5, 4/5
и ошибки практически не меняются: у delayed-policy устойчивое смещение,
не только недостаточное время достижения. Снимки движения и learning curves
PNG/SVG экспортированы и просмотрены.

Headless CLI полной механики прошёл 12 s с целями +80/−80/0 mm, без
падения, полная position RMSE с переходами **17,906 mm**. Это другая метрика,
чем установившаяся ошибка в таблице выше. Реальный GUI открыт/закрыт за
0,3 s, без segmentation fault, exit 0. Внутри sandbox X11 не открывался;
повтор вне sandbox успешен. После проверки API надписи исправлен enum
`mjGRID_TOPCENTER` на `mjGRID_TOP`. Для уже обученных весов явно проверено
единственное отличие исходника и выполнена совместимость metadata:
`overlay_compatibility.json`, исходник/metadata до правки сохранены,
фактический training source hash не потерян, веса всех 12 файлов неизменны.
После этого повторены строгие загрузки/метрики и полный набор тестов.

Первый незавершённый подготовительный запуск находится отдельно в
`runs/vertical_ppo_setup_incomplete/` и в результаты не включён.
Новых assets и внешних сервисов нет, generated weights остаются ignored.

## Исходный предиктор: бюджет ×10 и меньшие lr — 2026-11-06

```bash
.venv/bin/python3 -m pytest -q tests/test_vertical_tuning.py
.venv/bin/python3 scripts/train_vertical_budget.py > runs/vertical_budget.log 2>&1
.venv/bin/python3 scripts/evaluate_vertical_tuning.py --run runs/vertical_budget > runs/vertical_budget_evaluation.log 2>&1
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/plot_vertical_budget.py
.venv/bin/python3 -m py_compile scripts/train_vertical_budget.py scripts/plot_vertical_budget.py
git diff --check
```

Команды exit 0. Regression контракта/диапазона/задержки — **4 passed**, 1,63 s.
Полный набор повторно не запускался: изменены только скрипты оркестрации,
графиков и документация; runtime физики, наблюдений, сети и сериализации
сохранён. Последний полный набор — **125 passed** при коротком подборе параметров.

Все четыре исходные механики, данные и split сохранены, без истории.
Tanh 128×128, линейный выход, переход 20 ms, BPTT 60 ms. Бюджет каждой
стадии увеличен 1000/2000 → **10 000/20 000 updates**. Проверены lr
0,001 / 0,0003 / 0,0001 / 0,00003. Обучение с начала, те же seed и minibatch;
не заявляется возобновление старого optimizer. **32 checkpoint**, суммарно
**480 000 updates**, **1036,65 s** CPU с двумя torch-потоками. Артефакты —
`runs/vertical_budget/summary.json`, status `complete`.

По validation выбрана пара **0,001 / 0,0003**. Общий итоговый score
**0,86834** против **0,86934** длинного контроля с lr 0,001 — преимущество
меньшего lr всего **0,114%**. Относительно короткого контроля score улучшился
**1,08144 → 0,86834**, на **19,70%**. При неизменном новом родителе BPTT
score за первые 2000 updates **0,94014**, за все 20 000 — **0,86834**.
Лучшие BPTT updates выбранной пары: **0 / 0 / 15500 / 0**; ноль означает
сохранение подготовки, все запланированные updates выполнены.

После фиксации выбора test velocity RMSE на 60 ms —
**0,488 / 0,606 / 0,522 / 0,630 mm/s** против короткого Tanh/linear контроля
**0,695 / 0,854 / 5,999 / 2,795 mm/s**. На 20 ms —
**0,240 / 0,553 / 0,260 / 0,375 mm/s**. Основная часть улучшения получена
и при прежнем lr, за счёт бюджета. Модель остаётся предиктором, не actor.

Независимая проверка `independent_evaluation.json`, status `passed`:
строго загружены **32 checkpoint**, повторены train/validation/test,
train-only нормировки, все решения выбора. Проверен native replay и
Jacobian всех **16** итоговых кандидатов на шести состояниях каждой механики;
до 40 ms производная по новой команде у задержанных вариантов строго нулевая.
У полной механики относительная ошибка Jacobian на 60 ms **0,5247** —
лучше прежних **0,6034**, но всё ещё велика для вывода о пригодности к actor.

Первые 1000 updates всех **16** одношаговых ветвей точно совпали с прежними
журналами: `budget_comparison.json`. Четыре графика PNG/SVG экспортированы
и просмотрены. При lr 0,0001 best механики 3 на последнем BPTT update;
плато не установлено. У механик 2/4 дальнейший BPTT ухудшает validation,
несмотря на снижение train loss. Методика, таблицы и ограничения —
[vertical-budget.md](vertical-budget.md).

## Исходный предиктор: подбор learning rate и активаций — 2026-11-06

```bash
.venv/bin/python3 -m pytest -q tests/test_vertical_tuning.py
.venv/bin/python3 -m myo.train_vertical_tuning > runs/vertical_tuning.log 2>&1
.venv/bin/python3 -m pytest -q
.venv/bin/python3 scripts/evaluate_vertical_tuning.py > runs/vertical_tuning_evaluation.log 2>&1
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/plot_vertical_tuning.py
.venv/bin/python3 -m py_compile src/myo/train_vertical_tuning.py scripts/evaluate_vertical_tuning.py scripts/plot_vertical_tuning.py tests/test_vertical_tuning.py
git diff --check
```

Все команды exit 0. Новые случаи **4 passed**; полный набор вне sandbox:
**125 passed**, два прежних предупреждения Gymnasium/SB3, 24,01 s.
Проверены одинаковые линейные начальные веса между активациями, нулевая
производная по новой команде до 40 ms и ненулевая после, сохранение диапазона
и единичной производной Tanh-головы в нуле, перенос и несовместимость checkpoint.

Восстановлены исходные входы минимального стенда без истории: q/v, длины
волокон, активации в механике 4 и дошедшая команда. Четыре прежние механики,
данные и физика не менялись. Нейронный шаг 20 ms; основные горизонты 20/60 ms,
BPTT на трёх переходах. 120/400 ms — дополнительные проверки, не критерий выбора.

Последовательно проверены пять lr каждой стадии, три скрытые активации
и три выходные головы при выбранной скрытой активации. **72 checkpoint**,
199,29 s CPU с двумя torch-потоками, один seed. `runs/vertical_tuning/summary.json`,
status `complete`. По общему validation выбраны lr **0,001 / 0,001**,
**SiLU / Tanh с train-диапазоном**. Test рассчитан после фиксации выбора.
Меньшие lr при этом бюджете общего выигрыша не дали; оптимальность верхней
границы проверенной сетки не установлена.

Независимая проверка — `independent_evaluation.json`, status `passed`:
72 checkpoint, train-only нормировки и выходные границы, все сохранённые
ошибки и решения выбора повторены. Native replay на шести состояниях
каждой механики совпал побитово; измерены Jacobian 36 кандидатов, у задержанных
вариантов влияние текущей команды до 40 ms строго нулевое. Графики PNG/SVG
экспортированы и просмотрены.

У SiLU-linear test velocity RMSE на 60 ms по четырём механикам —
**0,337 / 4,981 / 5,972 / 1,418 mm/s**, у нового Tanh-linear контроля —
**0,695 / 0,854 / 5,999 / 2,795 mm/s**. Общего улучшения нет. Обычный
выход Tanh ограничивает 13–38% отдельных train-каналов полной механики;
масштабированный вариант представляет все train-значения, но не улучшает
test устойчиво. Формальный общий validation winner не рекомендуется
автоматически вместо всех исходных моделей.

У нового Tanh-контроля механик 3/4 выбран update 0 BPTT, несмотря на
уменьшение полной train loss при дальнейших updates. Прежние checkpoints,
обученные с BPTT 120 ms, на test остаются точнее новых в задержанных
механиках. Это не изоляция только горизонта: отличаются также порядок
minibatch и критерий выбора. Сравнение параметров выполнено, но оно не
установило достаточность короткого горизонта или пригодность для actor.
Методика, таблицы, validation-разбор и ограничения — в [vertical-tuning.md](vertical-tuning.md).

## Аудит параметров обучения предиктора — 2026-11-06

```bash
.venv/bin/python3 scripts/audit_tissue_training.py > runs/tissue_training_audit.log 2>&1
.venv/bin/python3 -m py_compile scripts/audit_tissue_training.py
git diff --check
```

Команды exit 0. Результат `runs/tissue_sampling/training_audit.json`, status
`passed`. На копиях выбранных warm-up весов точно воспроизведены норма
градиента и validation score первого исходного BPTT-update всех **12 ветвей**.
Проверены пять learning rate: 0,001 / 0,0003 / 0,0001 / 0,00003 / 0,00001.
Исходные веса, датасет и код обучения не менялись; новых trained checkpoints
нет. Unit-тесты не добавлялись и полный набор повторно не запускался:
изменены только диагностический скрипт и документация, последняя проверка
runtime — 121 passed в следующем разделе.

В полной механике на 1 ms первый update при lr 0,001 повышает loss на всей
train-выборке **0,06210 → 7,38660**, validation score **0,49518 → 2,72564**.
Норма градиента **3,73**, clipping до 10 не действует. С той же начальной
точки и minibatch при lr 0,00003: **0,05045 / 0,48330**. Это локальная
диагностика размера шага, не утверждение об оптимальном постоянном lr.

Измерены вклады групп каналов в loss, насыщение скрытых Tanh на собственной
развёртке, входная коррелированность и native one-step validation на всех
фазах команды. У полной модели 1 ms в первом/втором Tanh насыщено
**16,21% / 5,56%**, у 20 ms — **1,37% / 0,51%**. Доля активаций в общей
validation loss у 20/1 ms — около 44%, но они не входят в критерий выбора.
Выборка warm-up соответствует 53,33 / 13,33 / 2,67 эквивалентным проходам
по окнам при 20/5/1 ms. Все эти наблюдения требуют раздельных проверок причин.
Параметры, таблицы и приоритеты — в [tissue-training-review.md](tissue-training-review.md).

## Частота ощущений 20/5/1 ms при командах раз в 20 ms — 2026-11-06

```bash
.venv/bin/python3 -m pytest -q tests/test_tissue_sampling.py
.venv/bin/python3 -m myo.train_tissue_sampling > runs/tissue_sampling.log 2>&1
.venv/bin/python3 -m pytest -q
.venv/bin/python3 scripts/evaluate_tissue_sampling.py > runs/tissue_sampling_evaluation.log 2>&1
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/plot_tissue_sampling.py
.venv/bin/python3 -m py_compile src/myo/train_tissue_sampling.py scripts/evaluate_tissue_sampling.py scripts/plot_tissue_sampling.py tests/test_tissue_sampling.py
git diff --check
```

Две новые проверки: **2 passed**. После исправления выборки фаз warm-up полный
набор вне sandbox: **121 passed**, два прежних предупреждения Gymnasium/SB3,
31,03 s. Первый полный запуск внутри sandbox: 120 passed и один отказ из-за
запрета loopback UDP-сокета; тот же тест отдельно вне sandbox также passed.
Остальные перечисленные команды exit 0. Окружение — Python 3.12.16.

Повторно собраны физические подшаги всех четырёх механик. Команды эвристики
сохраняют период 20 ms, физика — 1 ms, транспорт — 40 ms при его наличии.
Проверены совпадение с исходным движением, причинная история/reset, все фазы
коротких переходов, неизменная очередь и градиент через собственные прогнозы.
Ни глобальная скорость, ни истинные будущие ощущения не подставляются в сеть.

Основной запуск — `runs/tissue_sampling/summary.json`, status `complete`,
**431,31 s** CPU, два torch-потока. Получены 12 прямых моделей на 20 ms
и 12 моделей развёртки до 60 ms; независимый скрипт строго проверил **32
checkpoints**, train-only нормировки, validation/test ошибки, выбор best и
критерий остановки. `independent_evaluation.json`, status `passed`.
Короткий первый запуск прерван из-за выборки warm-up только в фазе границы
команды; основной запуск использует все фазы, черновые числа исключены.

В полной механике прямой test RMSE длины волокна на 20 ms с историей через
20/5/1 ms — **0,05047 / 0,07574 / 0,10776 mm**. После собственной развёртки
и BPTT ошибка волокна на 60 ms — **0,07381 / 0,08235 / 0,22637 mm**.
Уплотнение сырых кадров не дало общего улучшения. Однако причинная оценка
текущей скорости пути по длинам стала точнее: RMSE
**1,24913 / 0,26296 / 0,000867 mm/s**. Native-скорость использовалась только
как эталон этой отдельной диагностики. Значит, более частые измерения
содержат полезную информацию, которую выбранная MLP не использовала для
общего улучшения прогноза.

В шести native-пробах полной механики ошибка производной местных скоростей
40–60 ms по новой команде относительно нулевого отклика —
**0,88399 / 0,72419 / 0,47147**. Короткий переход улучшил эту метрику
при ухудшении траекторий; ни то ни другое не является доказательством
пригодности для обучения actor. Один seed, одинаковый бюджет обновлений,
разная стоимость графа и число параметров первого слоя; достигнутый предел
точности не установлен. Графики PNG/SVG экспортированы и просмотрены.
Полная методика, все четыре механики, сухожилия, baselines и ограничения —
в [tissue-sampling.md](tissue-sampling.md). Контроллер не обучался.

## Качество, плато и размер локального предиктора — 2026-11-05

```bash
.venv/bin/python3 -m pytest -q tests/test_tissue_capacity.py
.venv/bin/python3 -m myo.train_tissue_capacity > runs/tissue_capacity.log 2>&1
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m pytest -q tests/test_telemetry.py::test_command_receiver_ignores_old_episode_and_timeout_only_requests_relaxation
.venv/bin/python3 scripts/audit_tissue_quality.py > runs/tissue_quality_audit.log 2>&1
.venv/bin/python3 scripts/evaluate_tissue_capacity.py > runs/tissue_capacity_evaluation.log 2>&1
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl .venv/bin/python3 scripts/plot_tissue_capacity.py
.venv/bin/python3 -m py_compile src/myo/train_tissue_capacity.py scripts/audit_tissue_quality.py scripts/evaluate_tissue_capacity.py scripts/plot_tissue_capacity.py tests/test_tissue_capacity.py
git diff --check
```

Новый regression: **1 passed**, все добавленные скрытые слои получают градиент,
сохраняются/восстанавливаются точно; несовместимая архитектура отвергается.
Полный набор в sandbox: **118 passed, 1 failed**, два прежних предупреждения
Gymnasium/SB3. Единственный отказ — запрет создания loopback UDP-сокета;
тот же тест отдельно вне sandbox: **1 passed**. Остальные команды exit 0.

Исходные 12 небольших моделей проверены против удержания длин и постоянных
местных скоростей, затем на native-изменении команд: шесть snapshot на каждую
из четырёх механик, ±0,005 первой команды, шесть шагов по 20 ms. Replay
состояний совпал до float32. `runs/tissue_predictor/quality_audit.json`,
status `complete`. Прогнозы через 120 ms лучше простых baseline, но влияние
команд существенно менее точно: в полной механике у исходной `derivative_loss`
Jacobian RMSE составляет 1,1645 ошибки нулевого Jacobian.

Завершены 12 новых моделей: четыре механики × 128×128 / 512×512 /
512×512×256, фиксированная ветвь `derivative_loss`. 1000 one-step, до 6000
BPTT; остановка при 1600 обновлениях без значимого улучшения validation ≥1%,
не ранее update 2000. Все завершились этим критерием, 612,33 s суммарно
на CPU, одновременно шли проверки/аудит. Small остановки: 3800/3600/3000/4400;
это не предел точности и не прежний фиксированный бюджет.

Независимо проверены **36 checkpoints**, train-only нормировки, все сохранённые
pooled/test ошибки и stop criterion. Small до update 2000 побитово совпала
с прежней ветвью во всех четырёх механиках; native-эталоны совпали точно.
`runs/tissue_capacity/independent_evaluation.json`, status `passed`.

В полной механике deep против продолженной small: test RMSE местных скоростей
на 120 ms 0,10539 → 0,08998; относительная ошибка native Jacobian
0,9891 → 0,7667, но base test MSE 0,034337 → 0,050473.
Все результаты одного seed; общий достаточный размер сети не установлен.
Никакая policy не обучалась. Графики PNG/SVG просмотрены, методика и полные
таблицы — в [tissue-quality-and-capacity.md](tissue-quality-and-capacity.md).

## Локальные ощущения тканей и их производные — 2026-11-05

```bash
.venv/bin/python3 -m pytest -q tests/test_tissue_predictor.py
.venv/bin/python3 -m myo.train_tissue_predictor > runs/tissue_predictor.log 2>&1
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m pytest -q tests/test_telemetry.py::test_command_receiver_ignores_old_episode_and_timeout_only_requests_relaxation
.venv/bin/python3 scripts/evaluate_tissue_predictor.py
.venv/bin/python3 -m py_compile src/myo/train_tissue_predictor.py scripts/evaluate_tissue_predictor.py tests/test_tissue_predictor.py
git diff --check
```

Новые проверки **3 passed**. Полный набор в sandbox: **117 passed, 1 failed**,
два прежних предупреждения Gymnasium/SB3. Единственный отказ — запрещённое
создание loopback UDP-сокета; тот же тест отдельно вне sandbox: **1 passed**.
Другие команды exit 0. Проверены независимые волокно/сухожилие, исключение
глобальной скорости, причинные производные/reset по эпизодам, их согласованность
с собственными прогнозами, сдвиг истории с градиентом, точная задержка
и контракт сохранения. Проверка структурного градиента сделана вне насыщения
Tanh, а не на почти постоянных нормировках четырёхшагового fixture.

Обучены 12 ветвей (четыре механики × три условия), 1000 one-step
и 2000 BPTT обновлений; 179,92 s с подготовкой сенсоров и оценками, status
`complete`. Исходная физика и прежние checkpoints не менялись.
Глобальные q/qvel используются только в старом источнике данных; граница
`native_to_tissues` преобразует геометрию в местные относительные длины,
qvel не использует. Новая сеть и loss не получают глобальные координаты.

Независимо загружены **24 checkpoints**, сверены train-only нормировка,
общие масштабы ошибок и точное воспроизведение всех pooled/test метрик.
Прежняя 60 ms модель оценена в тех же локальных каналах без подстановки
будущих истинных ощущений. `runs/tissue_predictor/independent_evaluation.json`,
status `passed`.

Для механики 4 test base MSE: только относительные длины/активации 0,024355,
также производные на входе 0,037697, также ошибка производных 0,038079.
На 120 ms последняя ветвь улучшила ускорение волокон (0,03839 → 0,01724 m/s²),
но ухудшила скорость (1,248 → 2,373 mm/s) и длину (0,1119 → 0,1977 mm).
Это компромисс метрик одного бюджетного запуска, не успешный контроллер
и не установленная точность его будущих градиентов. Полный отчёт —
[tissue-predictor.md](tissue-predictor.md).

## Явная история вертикального предиктора: 60 ms — 2026-11-05

```bash
.venv/bin/python3 -m myo.train_vertical_history --history-steps 3 --output-dir runs/vertical_sensory_history_60ms > runs/vertical_sensory_history_60ms.log 2>&1
.venv/bin/python3 -m pytest -q tests/test_vertical_history.py
git diff --check
```

Обучение завершено, exit 0, status `complete`, 89,23 s, восемь ветвей.
Проверки истории: **2 passed**, включая причинное заполнение трёх кадров,
сдвиг собственных прогнозов с градиентом, задержку и checkpoint contract.
Код физики/наблюдений/обучения и прежние checkpoints не менялись;
применён существующий параметр `--history-steps 3`.
Все четыре контрольных BPTT checkpoint побитово совпали с исходными;
SHA256 эпизодов совпали с опытом 200 ms.
Повторный аудит через `load_history`, `fit_normalization`, `normalized_errors`
строго загрузил 16 checkpoints, проверил train-only нормировку и точно
воспроизвёл все сохранённые нормированные ошибки. Результат:
`runs/vertical_sensory_history_60ms/artifact_check.json`, status `passed`.

В механике 4: test MSE 0,021743 против 0,032666 при 200 ms и 0,040623
без истории; RMSE скорости на 20/60/120/400 ms:
0,333/0,703/0,876/1,334 mm/s. На 20 ms точнее окно 200 ms,
на 400 ms — исходная модель без истории. Во всех механиках test MSE
короткого окна ниже длинного; в 1–3 отсутствие истории всё ещё лучше.
Полные таблицы и команда — в [vertical-sensory-history.md](vertical-sensory-history.md).

## Фиксированный рекуррентный резервуар вертикального предиктора — 2026-11-05

```bash
.venv/bin/python3 -m pytest -q tests/test_vertical_reservoir.py
.venv/bin/python3 -m myo.train_vertical_reservoir > runs/vertical_reservoir.log 2>&1
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m pytest -q tests/test_telemetry.py::test_command_receiver_ignores_old_episode_and_timeout_only_requests_relaxation
.venv/bin/python3 scripts/audit_vertical_reservoir.py
.venv/bin/python3 -m py_compile src/myo/train_vertical_reservoir.py tests/test_vertical_reservoir.py
git diff --check
```

Новые проверки: **3 passed**. Полный набор в sandbox: **114 passed, 1 failed**,
два прежних предупреждения Gymnasium/SB3. Единственный отказ — создание локального
UDP-сокета, `PermissionError: Operation not permitted`; тест отдельно вне
sandbox: **1 passed**. Остальные команды exit 0. Проверены границы/reset эпизодов,
причинная подготовка памяти, обновление по собственному прогнозу с градиентом,
радиус 0,8, фиксированность матриц при optimizer step, транспортная задержка,
точная загрузка обоих readout и отказ при несовместимой конфигурации памяти.

Обучены 8 ветвей: четыре механики × MLP/linear readout, по 1000 one-step
и 2000 BPTT обновлений. `runs/vertical_reservoir/summary.json`, status `complete`,
190,59 s с оценками и подготовкой памяти; рядом выполнялись тесты.
Физика, исходный trainer и предыдущие checkpoints не менялись.
Фиксированный резервуар: 256 нейронов, радиус 0,8, плотности 0,1, seed 0.
Сверены SHA256 исходных данных и строго загружены прежние baseline модели.

Независимый аудит: **16 checkpoints**, совпали с исходной инициализацией
`W_res/W_in`, train-only normalization и все сохранённые нормированные ошибки.
В полной механике test MSE: baseline 0,040623; явная история 0,032666;
reservoir + MLP 0,072564; reservoir + linear 0,451688. В механиках 1–3
резервуар также хуже по test loss. Бюджетный опыт, один seed; сходимость,
точность градиентов по native-динамике и успешный actor не установлены.
Полные результаты и отличия от статьи — в [vertical-reservoir.md](vertical-reservoir.md).

## История ощущений вертикального предиктора — 2026-11-05

```bash
.venv/bin/python3 -m pytest -q tests/test_vertical_history.py
.venv/bin/python3 -m myo.train_vertical_history > runs/vertical_sensory_history.log 2>&1
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m pytest -q tests/test_telemetry.py::test_command_receiver_ignores_old_episode_and_timeout_only_requests_relaxation
.venv/bin/python3 -m py_compile src/myo/train_vertical_history.py tests/test_vertical_history.py
git diff --check
```

Новые проверки истории: **2 passed**. Полный набор в sandbox: **111 passed,
1 failed**, два прежних предупреждения Gymnasium/SB3. Единственный сбой —
`PermissionError: Operation not permitted` при создании loopback UDP-сокета;
тот же тест отдельно вне sandbox: **1 passed**. Остальные команды exit 0.
Проверены причинная история внутри эпизода и её reset, сдвиг собственных
прогнозов с градиентом, сохранение задержки 40 ms, общие масштабы loss,
точная загрузка и отказ при несовместимой длине истории/схеме.

Обучены 8 ветвей для четырёх механик: текущие ощущения / также 200 ms
истории, 1000 one-step + 2000 BPTT обновлений на ветвь. Выполненный запуск
`runs/vertical_sensory_history/summary.json`, status `complete`, 104,27 s.
Данные не пересобирались, модель физики и исходный trainer не менялись.
Все выбранные контрольные BPTT веса побитово совпали с исходным опытом.
Внешний повторный аудит через `load_history`, `fit_normalization` и
`normalized_errors` проверил **16 checkpoints**, train-only нормировку
и точное воспроизведение всех сохранённых нормированных ошибок;
также совпали все 8 one-step/BPTT контрольных checkpoints с исходными.
Результат аудита — `runs/vertical_sensory_history/artifact_check.json`.

Результат смешанный: в механике 4 одношаговая test RMSE скорости
0,506 → 0,263 mm/s, но на 120 ms — 1,107 → 1,302,
на 400 ms — 1,005 → 1,969 mm/s. Шестишаговая нормированная test loss
0,040623 → 0,032666. В механиках 1–3 test-ошибки хуже.
Это один seed с прежним бюджетом, не доказательство общей пользы/бесполезности
истории и не успешный обученный контроллер. Полные таблицы и методика —
[vertical-sensory-history.md](vertical-sensory-history.md).

## Поканальный аудит нормированного loss вертикального предиктора — 2026-11-05

```bash
.venv/bin/python3 scripts/audit_vertical_loss.py --run runs/vertical_predictor
.venv/bin/python3 -m py_compile scripts/audit_vertical_loss.py
git diff --check
```

Оценка без optimizer и без изменения весов, exit 0. Строго загружены четыре
выбранных BPTT checkpoints, использованы сохранённые train-only delta scales.
Собраны все окна H=6, stride=1: train 2370, validation/test по 1185.
Проверено совпадение среднего поканальных MSE с точной формулой training loss
с допуском `1e-6 * max(1, loss)`; float64 использован только для агрегации отчёта.
Дополнительно сохранён пересчёт по прежним 231 test-окну, H=20, stride=5.

Результаты в `runs/vertical_predictor/normalized_errors.json` и таблицах
[vertical-predictor.md](vertical-predictor.md). Общие loss train/validation/test:
вариант 1 — 0,001421/0,309415/0,019265;
вариант 2 — 0,014465/1,061662/0,026558;
вариант 3 — 0,016117/0,937185/0,024247;
вариант 4 — 0,017745/0,480095/0,040623.
На test варианта 4 доли loss: скорость 44,96%, две активации 48,20%, положение
и две длины вместе 6,84%. Значительный разрыв validation/test установлен,
его причина здесь не исследовалась. Физика, модель и обучение не менялись.

## Цилиндры мышц и сухожилий вертикального стенда — 2026-11-05

Полноразмерные пути заменены декорациями: на каждую тягу отдельные цилиндры
волокна и сухожилия, независимые длины, радиусы по постоянному объёму,
палитры активации/нагрузки локтевого демо, легенды и две панели телеметрии.
Физический файл `vertical_muscles.py` и алгоритм предиктора не менялись.
Описание геометрии и численной точности — в [vertical-predictor.md](vertical-predictor.md).

```bash
.venv/bin/python3 -m pytest -q tests/test_vertical_viewer.py tests/test_vertical_visualization.py tests/test_vertical_muscles.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -X faulthandler -m myo.vertical_muscles --variant 4 --view
.venv/bin/python3 -m py_compile src/myo/vertical_visualization.py src/myo/vertical_viewer.py tests/test_vertical_visualization.py tests/test_vertical_viewer.py
git diff --check
```

Целевые тесты: **5 passed in 2.81 s**; полный pytest с разрешённым loopback
UDP: **110 passed, 2 warnings in 45.35 s**, exit 0. Предупреждения SB3 прежние.
Новая проверка использует два разных равновесия мышцы/сухожилия при одном
положении груза: граница тканей и радиусы должны меняться. Дополнительно
проверены постоянство `r²L`, фактические размеры MuJoCo-цилиндров, фиксированные
палитры и постоянство длины/радиуса нерастяжимого сухожилия варианта 1.
Сохранились проверки неизменности физики и ожидания GUI-потока при закрытии.

Через `MUJOCO_GL=egl` и `myo.vertical_visualization.render_frame()` построены
шесть PNG: варианты 1/4, шаги 25/100/150. Просмотрены кадры 4/25, 4/150 и
1/100; изменена подсветка renderer для читаемых цветов. Итоговые кадры
сохранены в `runs/vertical_tissue_qa/`. Первый PNG-export без PYTHONPATH
не нашёл PIL; повтор использовал уже имеющиеся plotting-зависимости в
`/tmp/myo-supervised-plot-deps`, установки зависимостей не было.

```bash
PYTHONPATH=/tmp/myo-supervised-plot-deps MUJOCO_GL=egl .venv/bin/python3 - <<'PY'
from pathlib import Path
from PIL import Image
from myo.vertical_muscles import VerticalConfig, VerticalMuscles, heuristic
from myo.vertical_visualization import render_frame
out = Path('runs/vertical_tissue_qa')
out.mkdir(exist_ok=True)
for variant in (1, 4):
    env = VerticalMuscles(VerticalConfig(variant=variant))
    env.reset()
    for step in range(151):
        if step in (25, 100, 150):
            Image.fromarray(render_frame(env)).save(out / f'variant_{variant}_step_{step}.png')
        env.step(heuristic(env, 'steps'))
PY
```

В реальном GUI с локальным дисплеем проверены варианты 1–4 по 0,4 s и
вариант 4 на полные 12 s: все exit 0. Полный прогон сохранил прежние 600 шагов
и диапазон −0,04858581656431187…+0,05662575469207157 m. Результаты —
`runs/vertical_tissue_qa/gui_checks.json`.

Все восемь прежних checkpoints строго загрузились с текущими contracts;
переобучения не было. В `precision.json` подтверждены float32 и test-ошибки
положения варианта 4: 0,0117/0,1105/0,3204 mm на 20/120/400 ms. Соседний
интервал float32 около 0,1 m равен `7,45e-9 m`; это численная разрешающая
способность, а не обещание такой точности прогноза.

## Закрытие GUI вертикального стенда — 2026-11-05

Воспроизведён сообщённый пользователем segfault после завершения симуляции:

```bash
.venv/bin/python3 -X faulthandler -m myo.vertical_muscles --variant 4 --view --seconds 0.2
```

До исправления: **exit 139**. Faulthandler показал одновременные
`glfw.terminate` в основном потоке и `_launch_internal` / `simulate.destroy`
в GUI-потоке MuJoCo. `Handle.close()` только отправляет запрос на завершение;
публичный `launch_passive` создаёт daemon-поток и не предоставляет join.

`VerticalViewer` теперь владеет недетаченным GUI-потоком, отправляет запрос
закрытия и дожидается его `join()` до выхода Python. Сохранён штатный GUI
MuJoCo через внутренний launch target версии 3.17. Отдельные model/data
предотвращают перенос GUI-правок в физику; камера/копирование/цвета меняются
под viewer lock. Произвольной задержки после close нет.

На локальном `DISPLAY=:1` вне ограничений sandbox выполнены десять отдельных
коротких запусков по 0,2 s и обычный запуск без `--seconds` на 12 s:
**все exit 0**. Полный запуск дал прежние 600 шагов и диапазон
−0,04858581656431187…+0,05662575469207157 m. В отдельном процессе также
проверены запрос закрытия перед cleanup и исключение из эвристики — exit 0.
Лог: `runs/vertical_viewer_shutdown_check.json`.

```bash
.venv/bin/python3 -m pytest -q tests/test_vertical_viewer.py tests/test_vertical_muscles.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo.train_vertical_predictor \
  --output-dir runs/vertical_predictor --seconds 8 \
  --one-step-updates 1000 --bptt-updates 2000 \
  > runs/vertical_predictor.log 2>&1
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/plot_vertical_predictor.py --run runs/vertical_predictor
.venv/bin/python3 -m py_compile src/myo/vertical_viewer.py src/myo/vertical_muscles.py tests/test_vertical_viewer.py
git diff --check
```

Целевые тесты: **4 passed in 1.87 s**; полный pytest с разрешённым loopback
UDP: **109 passed, 2 warnings in 36.01 s**, exit 0. Предупреждения SB3 прежние.
Новая регрессия удерживает освобождение GUI в потоке и проверяет, что close
не возвращается раньше завершения cleanup; также имитирует GUI-изменение
положения и проверяет неизменность физического состояния.

Поскольку GUI entry point находился в хешируемом файле стенда, прежний набор
сохранён в `runs/vertical_predictor_before_viewer_fix/` вместе с восстановленным
и проверенным по hash исходником. Набор пересоздан в исходной папке за 68,88 s.
Физический и эвристический код до `main()` совпал точно; во всех четырёх
вариантах совпали состояния, действия, телеметрия, все tensors обоих checkpoints,
normalization и test/native-метрики. Новые checkpoints строго загружаются
с актуальными contracts. Проверка — `final_artifact_check.json`; графики
перестроены, exit 0. Прежние metadata вручную не подменялись.

## Минимальный вертикальный стенд, четыре предиктора — 2026-11-05

Реализован отдельный MuJoCo-стенд с двумя мышцами и грузом 1 kg; подробности,
параметры и ограничения — в [vertical-predictor.md](vertical-predictor.md).

```bash
.venv/bin/python3 -m myo.vertical_muscles --variant 4 --seconds 8
.venv/bin/python3 -m pytest -q tests/test_vertical_muscles.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo.train_vertical_predictor \
  --output-dir runs/vertical_predictor --seconds 8 \
  --one-step-updates 1000 --bptt-updates 2000 \
  > runs/vertical_predictor.log 2>&1
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/plot_vertical_predictor.py --run runs/vertical_predictor
.venv/bin/python3 -m py_compile src/myo/vertical_muscles.py src/myo/train_vertical_predictor.py scripts/plot_vertical_predictor.py tests/test_vertical_muscles.py
git diff --check
```

Headless-прогон варианта 4: 400 управляющих шагов, диапазон
−0,04415…+0,05663 m, exit 0. Целевые тесты: **3 passed in 1.36 s**;
полный pytest с разрешённым loopback UDP: **108 passed, 2 warnings in 31.01 s**,
exit 0. Предупреждения SB3 прежние. Обучение и построение графиков завершены,
exit 0. PNG траекторий и сравнения просмотрены. При первом запуске GUI
не проверялся; позднее его реальное закрытие исправлено и проверено выше.

На вариант: 12 эпизодов по 8 s, разделение целыми эпизодами 6/3/3,
96000 физических шагов. MLP `128×128`, 1000 one-step + 2000 BPTT обновлений;
последний этап начинается с выбранного validation one-step checkpoint.
Всего запуск занял 53,85 s на CPU. Никакая управляющая policy не обучалась.

Test BPTT, RMSE положения на 400 ms по вариантам 1–4:
**0,417/0,349/0,331/0,320 mm**; скорости:
**0,000965/0,000933/0,000885/0,001005 m/s**. Cosine native/model Jacobian скорости
на 60–120 ms: **0,990/0,992/0,995/0,938**. Шесть native-проб на вариант и
2880 физических шагов на каждый проверяемый checkpoint; при задержке влияние
до 40 ms нулевое. У варианта 4 ошибка Jacobian 0,157 против нулевого baseline
0,453 — величина чувствительности ещё заметно неточна.

В `final_artifact_check.json` отдельно подтверждены актуальные contracts,
загрузка обоих этапов, раздельные выборки, train-only normalization,
совпадение ручной развёртки с rollout и точная мгновенная активация там,
где она задана. Max-расхождение autograd с центральной разностью самой сети
в float64, epsilon `1e-5`: `2,58e-8/1,74e-8/1,63e-8/4,38e-9`.
Это не проверка анатомической достоверности и не готовый контроллер.

## Шаг 5/1 ms и история дошедших команд — 2026-11-05

Реализован опыт с неизменным физическим шагом 1 ms, задержкой 40 ms и
горизонтом BPTT 120 ms. Методика — в [dynamics-timestep.md](dynamics-timestep.md).
Позднее остановлен владельцем в пользу вертикального стенда: обучение всех
ветвей дошло до 2000, но оценка второй selection последней ветви прервана.
Статус артефактов — `stopped_by_user`, не полностью завершённый опыт.

```bash
.venv/bin/python3 -m pytest -q tests/test_dynamics_timestep.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo.train_dynamics_timestep \
  --reference runs/d1_dynamics_sensors \
  --output-dir runs/d1_dynamics_timestep \
  --updates 2000 --batch-size 16 --check-every 200 \
  > runs/d1_dynamics_timestep.log 2>&1
.venv/bin/python3 -m py_compile src/myo/dynamics_command_history.py src/myo/train_dynamics_timestep.py tests/test_dynamics_timestep.py scripts/plot_dynamics_timestep.py
git diff --check
```

Целевые проверки: **4 passed in 1.60 s**; полный pytest с разрешённым
loopback UDP: **105 passed, 2 warnings in 24.55 s**, exit 0.
Предупреждения SB3 прежние. Проверены реальные очереди на 5/1 ms, отсутствие
влияния до 40 ms, точный сдвиг истории дошедших команд, отсутствие будущих
данных, сохранение/проверка шага checkpoint и общие временные метрики.

Переиграны 72 эпизода, 216000 физических шагов. Все 20 физических каналов
на исходной сетке 20 ms совпали с прежними **точно**. Native-проверка меняет
возбуждение на одинаковые 20 ms во всех вариантах: 36 состояний,
51840 физических шагов; максимальное отличие эталонных Jacobian от прежнего
опыта `3,11e-14`, влияние до 40 ms нулевое.

## Скорость волокон и сила сухожилий на входе — 2026-11-05

Добавлены 12 входных признаков большой сети, без изменения физики и loss.
Методика и результаты — в [dynamics-sensors.md](dynamics-sensors.md).

```bash
.venv/bin/python3 -m pytest -q tests/test_dynamics_sensors.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo.train_dynamics_sensors \
  --reference runs/d1_dynamics_acceleration \
  --output-dir runs/d1_dynamics_sensors \
  > runs/d1_dynamics_sensors.log 2>&1
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/evaluate_dynamics_prediction.py \
  --run runs/d1_dynamics_sensors \
  --reference-run runs/d1_dynamics_acceleration \
  --run-label 'with sensors' --reference-label 'without sensors' \
  --comparison-title 'Wide predictor: fiber-velocity and tendon-force inputs' \
  --acceleration --plot > runs/d1_dynamics_sensors_evaluation.log 2>&1
.venv/bin/python3 -m py_compile src/myo/dynamics_sensors.py src/myo/train_dynamics_sensors.py scripts/evaluate_dynamics_prediction.py tests/test_dynamics_sensors.py
git diff --check
```

Первый полный pytest выявил нестабильность нового тестового примера:
случайная ненормированная сеть насыщала Tanh на силе в N, и проверяемый
градиент исчезал. Исправлена только фикстура: задана нормировка силы и
детерминированные ненасыщенные веса. Рабочие модель и trainer не менялись.
После исправления целевые проверки: **2 passed in 2.63 s**; полный pytest
с разрешённым loopback UDP: **101 passed, 2 warnings in 29.25 s**, exit 0.
Предупреждения SB3 прежние. Обучение и независимая оценка — exit 0.

Сеть `38→1024→1024→512→20`, 1624596 параметров, 2000 обновлений;
оба критерия validation выбрали одинаковый update 1900. Старые состояния,
действия, разделение эпизодов и общая normalization совпали точно;
новые статистики используют только train. Физический контракт неизменен.
Сила на 216 native-показаниях: RMSE 0,00141 N, max 0,00566 N.
36 native-проб, 2808 управляющих шагов; эталонные Jacobian совпали точно.
Hashes, совместимость и метрики проверены в `final_artifact_check.json`.

Test RMSE скорости на 120 ms: 0,40916→0,38375 rad/s, локального ускорения:
10,21920→9,99344 rad/s². На 400 ms ускорение хуже: 10,30039→10,48575.
Совокупная ошибка Jacobian скорости выросла 0,772→0,854; ускорения снизилась
37,746→35,380, но cosine последнего около 0,10. Это небольшой выигрыш части
прогнозов, не доказательство пригодности предиктора для обучения управления.
Новый actor не обучался, демо не переключалось.

## Ускорение вместо скорости в loss предиктора — 2026-11-05

Выполнен контролируемый опыт на той же большой MLP и исходной физике.
Описание цели, эквивалентности one-step и результатов —
в [dynamics-acceleration.md](dynamics-acceleration.md).

```bash
.venv/bin/python3 -m pytest -q tests/test_dynamics_prediction.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo.train_dynamics_acceleration \
  --reference runs/d1_dynamics_prediction_wide \
  --output-dir runs/d1_dynamics_acceleration \
  > runs/d1_dynamics_acceleration.log 2>&1
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/evaluate_dynamics_prediction.py \
  --run runs/d1_dynamics_acceleration \
  --reference-run runs/d1_dynamics_prediction_wide \
  --run-label 'acceleration loss' --reference-label 'velocity loss' \
  --comparison-title 'Wide dynamics model: velocity loss vs interval acceleration loss' \
  --acceleration --plot > runs/d1_dynamics_acceleration_evaluation.log 2>&1
.venv/bin/python3 -m py_compile src/myo/train_dynamics_acceleration.py scripts/evaluate_dynamics_prediction.py tests/test_dynamics_prediction.py
git diff --check
```

Целевая проверка: **4 passed in 1.27 s**. Добавлена регрессия: нормированные
одношаговые ошибки ускорения/скорости имеют одинаковые значения и градиенты,
но многошаговая ошибка локального ускорения не штрафует постоянный сдвиг
скорости на каждом шаге. Полный pytest с разрешённым loopback UDP:
**99 passed, 2 warnings in 26.12 s**, exit 0; предупреждения SB3 прежние.
Обучение и независимая оценка завершены, exit 0.

Один новый запуск BPTT: 2000 обновлений, та же архитектура 1612308 параметров,
инициализация, данные и normalization. One-step checkpoint переиспользован,
поскольку замена его loss математически тождественна. Критерии validation
по ускорению и по скорости выбрали одинаковые веса на update 1600. Нового
сбора эпизодов нет. 36 native-проб добавили 2808 управляющих шагов;
native Jacobian точно совпал с предыдущим опытом, исходники физики не менялись.

RMSE ускорения в интервалах 0–20/100–120/380–400 ms:
10.397/10.219/10.300 rad/s² против 10.006/10.154/10.412 у прежнего BPTT.
RMSE скорости в концах этих интервалов .20794/.40916/.43273 rad/s против
.20012/.33813/.36360. Устойчивого выигрыша по ускорению нет, скорость хуже.
Совокупная RMSE Jacobian скорости уменьшилась 1.121→.772; Jacobian ускорения
60.915→37.746, но cosine последнего только .111, знак значимых элементов
совпадает примерно в 60%. Эти производные недостаточно точны для доверенного
обучения actor. Готовый контроллер, польза переноса признаков и верхний уровень
в этом опыте не проверялись.

## Большой предиктор `1024×1024×512` — 2026-11-05

Повторён предыдущий опыт с большой MLP, без изменения физики, данных и бюджета.
Полная методика и результаты — в [dynamics-prediction.md](dynamics-prediction.md).

```bash
.venv/bin/python3 -m pytest -q tests/test_dynamics_prediction.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo.train_dynamics_wide \
  --dataset runs/d1_dynamics_prediction \
  --output-dir runs/d1_dynamics_prediction_wide \
  --hidden-layers 1024 1024 512 \
  > runs/d1_dynamics_prediction_wide.log 2>&1
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/evaluate_dynamics_prediction.py \
  --run runs/d1_dynamics_prediction_wide \
  --reference-run runs/d1_dynamics_prediction --plot \
  > runs/d1_dynamics_prediction_wide_evaluation.log 2>&1
.venv/bin/python3 -m py_compile src/myo/train_dynamics_wide.py scripts/evaluate_dynamics_prediction.py tests/test_dynamics_prediction.py
git diff --check
```

Целевая проверка: **3 passed in 1.25 s**. Добавлена регрессия сохранения всех
трёх скрытых слоёв и точного восстановления выходов, отказа при несовпадении
базовой реализации и сохранения загрузки прежних малых checkpoints.
Полный pytest с разрешённым loopback UDP: **98 passed, 2 warnings in 23.98 s**,
exit 0, предупреждения SB3 прежние. Обучение и независимая оценка — exit 0.

Данные и native snapshots скопированы побайтно; train/validation/test те же
36/18/18 эпизодов. Обе ветви имеют по 1612308 параметров и 2000 обновлений.
Выбор по validation дал updates 1900/1800. Повтор 36 native-проб выполнил
2808 управляющих шагов; эталонные Jacobian совпали с первым опытом **точно**.
Одинаковая normalization и неизменная физика проверены отдельно, старые
checkpoints по-прежнему загружаются с исходными hashes.

RMSE test-скорости one-step на 20/120/400 ms: .20421/.40269/.40907 rad/s;
BPTT .20012/.33813/.36360. Большая BPTT лучше малой по этим значениям,
но совокупная RMSE Jacobian выросла .563→1.121, cosine упал .534→.277.
Особенно неверен первый отклик на 60 ms: противоположное направление в 11/18
состояний с заметным влиянием. На 120 ms направления улучшились: 3/28 таких
ошибок против 8/28 у малой BPTT; общий вывод не означает ухудшения всех горизонтов.

Дополнительная численная проверка фиксированной большой сети в float64
(шесть test-состояний, epsilon `1e-5`, без optimizer): RMSE autograd против
центральной разности `2.94e-8`/`1.27e-8`. Она подтверждает корректность
производных сети, но не точность модели физики. Результаты этой проверки
и hashes находятся в `autograd_check.json`/`final_artifact_check.json`.
Польза переноса признаков и успешное обучение actor через такую модель
не проверялись; большой предиктор не является новым контроллером демо.

## Одношаговая динамика и BPTT — 2026-11-05

Реализован и завершён отдельный опыт action-conditioned dynamics; методика,
состав состояния и ограничения — в [dynamics-prediction.md](dynamics-prediction.md).
Исходная физика не менялась, fingerprint по-прежнему
`bc7cacd13f9bbe8dc777919fdd018a3e96a12ced614b3850f996c18af953e8e3`.
Предиктор не подменяет физику и не подключён к actor или обычному демо.

Выполнены:

```bash
.venv/bin/python3 -m pytest -q tests/test_dynamics_prediction.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo.train_dynamics_prediction \
  --output-dir runs/d1_dynamics_prediction \
  > runs/d1_dynamics_prediction.log 2>&1
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/evaluate_dynamics_prediction.py --plot \
  > runs/d1_dynamics_prediction_evaluation.log 2>&1
.venv/bin/python3 -m py_compile src/myo/dynamics_prediction.py src/myo/train_dynamics_prediction.py scripts/evaluate_dynamics_prediction.py tests/test_dynamics_prediction.py
git diff --check
```

Новые проверки: **2 passed in 1.24 s**. Проверены окна без пересечения reset,
независимость normalization от validation, отсутствие градиента до истечения
40 ms и его наличие после задержки при BPTT, потеря связи с ранним действием
при detach, точное восстановление checkpoint и отказ при несовместимом timestep.
Первый полный pytest в sandbox: 96 passed, 1 failed из-за запрета открыть
loopback UDP socket в существующем тесте. Повтор с разрешённым локальным
сокетом: **97 passed, 2 warnings in 28.12 s**, exit 0; предупреждения SB3 прежние.

Обучение и независимая оценка завершились с exit 0. 72 эпизода, 10800 физических
управляющих шагов, 0 срывов; 36 native-проб добавили 2808 шагов. По 2000
обновлений двух одинаково инициализированных MLP; в BPTT шесть применений модели
на обновление. Один seed, train/validation/test = 36/18/18 целых эпизодов.
RMSE test-скорости на 20/120/400 ms: one-step .22310/.42647/.42629 rad/s,
BPTT .23128/.41117/.43728; baseline постоянной скорости
.33978/1.31665/1.48426. BPTT не дал устойчивого преимущества по всем горизонтам.

Native-аудит показал недостаточную точность производных: после задержки знак
значимых производных совпадает лишь в 75.14/76.30%, общий cosine .516/.534.
На 120 ms у BPTT противоположное направление вектора производных в 8 из 28
состояний с заметным откликом. Нулевой отклик первых 40 ms заложен архитектурой,
а не выучен. Ошибки резко возрастают на переходах в нижнюю область движения.
Это первый ограниченный опыт, не доказательство сходимости, полезности переноса
признаков, готового нижнего уровня или надёжного градиента для actor.

## Ускорения на исходной задержанной/упругой механике — 2026-11-04

По новому решению владельца вместо окна rigid-actor реализовано и запущено
обучение на исходном `PreacherCurlEnv`. Сохранены 40 ms моторной задержки,
15/50 ms активации/деактивации, отдельное упругое сухожилие, fatigue proxy,
контакт, геометрия, масса и шаги времени. Веса rigid-опыта не переносятся.
Методика численного отклика и ограничение локального приближения — в
[разборе](controller-learning-review.md#ускорения-на-исходной-задержаннойупругой-механике--2026-11-04).

Выполнены:

```bash
.venv/bin/python3 -m pytest -q tests/test_delayed_acceleration.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo.train_delayed_acceleration \
  --output-dir runs/d1_delayed_acceleration \
  > runs/d1_delayed_acceleration.log 2>&1
```

Две новые проверки: **2 passed in 3.56 s**. Для холодного старта и после
200 ms управляющей истории проверены совпадение отклика с независимо
клонированной полной средой, нулевое влияние новой команды в первые 40 ms,
ненулевой Jacobian после задержки, соответствие градиента loss независимой
численной разности и неизменность исходных наблюдений/мышечных состояний.
Полный pytest: **93 passed, 2 warnings in 24.96 s**, exit 0; разрешён локальный
сокет для прежнего UDP loopback-теста, предупреждения SB3 прежние.

Первый запуск принудительно остановлен после второго измерения, 600 обновлений:
лучшая оценка 7.32058 против начальной 14.86343, **0/6**. На проверке первой/
второй линеаризации RMSE приближения составила 2.865/1.515 rad/s², максимальный
уход команд .208/.270. Финальный live-счётчик физических шагов прерванного
процесса не получен; его нельзя считать завершённым запуском. Численные
datasets и checkpoints сохранены; избыточные `.pkl` с полным `MjData` удалены.
Исходники первого опыта скопированы в его каталог как `v1_*.py`.

После замены snapshots на `mjSTATE_INTEGRATION` + мышцы/очереди/историю
выполнено **3 passed in 7.43 s**, включая независимое воспроизведение в
контактной позе 142°. Один pickle-снимок уменьшился примерно с 26 MB до
15694 bytes; новый Jacobian измеряется на той же исходной физике.
Повтор полного pytest после исправления: **94 passed, 2 warnings in 35.55 s**,
exit 0. Для продолжения добавлен контроль ухода команд после каждой эпохи
и фиксированные ранние/поздние учебные отсчёты вместо хранения соседних кадров.

```bash
.venv/bin/python3 -m myo.train_delayed_acceleration \
  --initial runs/d1_delayed_acceleration/best.zip --rounds 8 \
  --output-dir runs/d1_delayed_acceleration_local \
  > runs/d1_delayed_acceleration_local.log 2>&1
```

В продолжении исправлено ещё одно конкретное нарушение: на неизменном наборе
round2 после minibatch-обновления RMSE локальной модели выросла
.89324→1.07721 rad/s². После четырёх проверок/670 суммарных обновлений оно
остановлено. В финальной версии каждый шаг использует полный набор, beta1=0
и backtracking с проверкой уменьшения полной loss и ухода команд ≤.03.
Веса и состояние Adam при неудачной пробе восстанавливаются. Для неодинаковой
плотности кадров добавлены веса представляемого физического времени.
Исходники промежуточной версии сохранены как `v2_*.py` в её каталоге.

```bash
.venv/bin/python3 -m pytest -q tests/test_delayed_acceleration.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo.train_delayed_acceleration \
  --initial runs/d1_delayed_acceleration_local/last.zip --rounds 8 \
  --output-dir runs/d1_delayed_acceleration_descent \
  > runs/d1_delayed_acceleration_descent.log 2>&1
.venv/bin/python3 scripts/evaluate_delayed_acceleration.py \
  > runs/d1_delayed_acceleration_descent_evaluation.log 2>&1
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 runs/d1_delayed_acceleration_descent/plot_report.py
.venv/bin/python3 -m py_compile src/myo/delayed_acceleration.py src/myo/train_delayed_acceleration.py scripts/evaluate_delayed_acceleration.py tests/test_delayed_acceleration.py
git diff --check
```

Финальные проверки: **4 passed in 4.28 s**, включая регрессию уменьшения
полной loss в пределах измеренной окрестности.
Полный pytest: **95 passed, 2 warnings in 27.65 s**, exit 0. Компиляция и
`git diff --check` — exit 0. Для PNG/SVG используется ранее установленный
во временный `/tmp` Matplotlib, без изменения среды обучения.

Финальное обучение завершено exit 0, `experiment_budget_exhausted`,
**700.38 s / 8 раундов**. Исходный счётчик 670, 310 новых принятых обновлений,
best/last на 980: score **3.636899**, **3/6**; двух подтверждений 6/6 нет.
У всех 310 принятых шагов loss на своём неизменном наборе уменьшается,
максимальный уход .0296816. Финальная RMSE точной teacher-validation
.40756 rad/s², native-error на учебных probe .42507 rad/s², ошибка самого
линейного приближения .049324 rad/s².

Сбор — **12000 физических управляющих шагов** (648 учебных состояний за
все сборы, из них 144 используются в одном раунде, плюс 60 отдельных
проверочных состояний). Численные отклики — **99288 native-шагов управления**,
включая teacher-продолжения, измерение Jacobian и точные проверки;
13 оценок во время обучения — **15600 физических управляющих шагов**.
Это разные счётчики, не количество PPO samples. Сборы полные, без физических
аварий. Исходные векторы наблюдений — 773, future-меток в них нет.

Независимая проверка best после обучения: **14400 физических управляющих
шагов**, без обновлений или teacher-помощи actor. Из 20°/21° на 4 s — 3/6;
fresh-конечные углы 30.43–30.59°, capacity=.8 — 23.25–23.27°.
На 20 s — также 3/6: fresh 27.45–28.10°, capacity=.8 22.34–22.35°.
Эвристика на 20 s — 6/6, 29.994°, RMSE около .006°. Отчёт:
`runs/d1_delayed_acceleration_descent/independent_evaluation.json`.

Inline native-аудит двух terminal-состояний при trigger=1: **400 шагов
политики + 180 шагов копий**, без обучения. При capacity=.8 и угле 23.255°
RMSE будущего ускорения .71764 rad/s²; все производные по сгибателям
отрицательны, по разгибателям положительны. Полезный градиент после задержки
существует и в этом неудачном состоянии. Первые 40 ms имеют нулевой Jacobian.
Отчёт `terminal_credit_audit.json` хранит команды, ускорения и производные.

Финальный inline-аудит всех восьми datasets подтвердил непересечение episode
IDs с validation, средние временные веса 1, нулевой Jacobian первых 40 ms,
обычную строгую загрузку best/last, неизменность critic/нормализатора и ZIP.
Основной dynamics SHA-256:
`bc7cacd13f9bbe8dc777919fdd018a3e96a12ced614b3850f996c18af953e8e3`.
Kernel SHA-256:
`a9715668d76bb51d47f20a432c7058796f7e1859a8fb8c4e5d62f4306466176a`;
runner SHA-256:
`14d80ab31fd507e690e302b88903685863f6c7979bcab174b85c6f19f11510f3`.
Best/last ZIP SHA-256:
`dbe82cff2937cbb018222fd71ae667096ebb51d20eaee757681a2db9717f3a37`.
Полные хеши — в `final_artifact_check.json` того же каталога.

Графики — `comparison.png`/`.svg`. Веса и данные остаются игнорируемыми;
в основном demo и механике изменения не вводились. GUI-прогон этой сети
не выполнялся. Ненулевой корректный градиент и положительные результаты
fresh-случаев не означают освоения всех условий, других целей или защиты.

На момент запуска никакого успешного нового motor policy не заявлялось.
Фактические результаты сохраняются в `runs/d1_delayed_acceleration/summary.json`
и оценках по раундам. Физический fingerprint и старые checkpoints сохранены;
новые веса используют исходную физику и обычный v2 runtime-контракт.

## Два supervised-опыта: команды и ускорения — 2026-11-04

Реализован и выполнен `myo.supervised_actuator`: два одинаковых исходных
actor на диагностической `instant_rigid` механике. Без задержки, с мгновенной
активацией и нерастяжимым сухожилием; основной привод D1 не менялся. Различие
ветвей — MSE мышечных команд либо MSE мгновенного физического ускорения от
этих команд относительно ответа эвристики в том же состоянии. Во втором
случае градиент проходит через силы и механический ответ, без мышечного
MSE, критика, PPO/GAE или backprop через время. Детали и ограничения — в
[разборе обучения](controller-learning-review.md#обучение-по-командам-и-ускорениям-эвристики--2026-11-04).

Выполнены:

```bash
.venv/bin/python3 -m pytest -q tests/test_supervised_actuator.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo.supervised_actuator \
  --output-dir runs/d1_supervised_actuator \
  > runs/d1_supervised_actuator.log 2>&1
.venv/bin/python3 scripts/evaluate_supervised_actuator.py \
  > runs/d1_supervised_actuator_evaluation.log 2>&1
.venv/bin/python3 -m pip install --no-cache-dir --target /tmp/myo-supervised-plot-deps matplotlib==3.10.7
PYTHONPATH=/tmp/myo-supervised-plot-deps MPLCONFIGDIR=/tmp/myo-supervised-mpl \
  .venv/bin/python3 scripts/plot_supervised_actuator.py
.venv/bin/python3 -m py_compile src/myo/supervised_actuator.py tests/test_supervised_actuator.py scripts/evaluate_supervised_actuator.py scripts/plot_supervised_actuator.py
git diff --check
```

Новые проверки: **2 passed in 1.59 s** — совпадение ускорения с независимым
расчётом сил/MuJoCo и градиента с центральной разностью при 30° и 142°.
На первом запуске обнаружено отсутствие `data.qM` в установленном MuJoCo;
до обучения исправлено на постоянную инерцию фиксированной 1-DOF модели
`model.dof_M0[0]`. Полный pytest после исправления: **91 passed, 2 warnings
in 37.13 s**, exit 0. Для прежнего UDP loopback-теста разрешён локальный
сокет; предупреждения SB3 прежние. Компиляция и `git diff --check` — exit 0.
Matplotlib 3.10.7 установлен только в `/tmp` для PNG/SVG, среда проекта не
изменялась. Первая попытка загрузки пакета не разрешила DNS в sandbox;
повтор с разрешённой сетью завершился успешно. Это только построение графика,
не другая среда обучения или физики.

Перед обучением inline Python собрал **1200 физических отсчётов** эвристики
и проверил метки ускорения. Диапазон −.2480…4.3633 rad/s², RMS .43138 rad/s²;
максимальное расхождение Torch-предсказания с MuJoCo **5.89e−6 rad/s²**.
Все эти состояния без активных ограничений; ограниченная поза отдельно
проверена тестом. Артефакт: `runs/d1_supervised_target_probe.npz`.

Основное обучение: exit 0, `both_stage_ready`, **330.30 s**. Задача 20→30°,
1 kg, trigger=.3/.6/1 × capacity=1/.8, фиксированный seed=0. Сбор:
1200 teacher-train + 1200 teacher-validation с началом 21° + по 1200
состояний двух собственных actor = **4800 физических управляющих шагов**.
Во втором раунде одинаковые 3600 учебных отсчётов переданы обеим ветвям.
Проверочные эпизоды целиком отделены от обоих учебных наборов.

После первого раунда 200 эпох обе ветви проходили 0/6. Во втором обе достигли
двух последовательных проверок 6/6 и остановлены. Лучшие checkpoints:

| Ветка | Обновлений Adam | RMSE команд на validation | RMSE ускорения на validation | Score 4 s |
| --- | ---: | ---: | ---: | ---: |
| `actions` | 5480 | .004106 | .79998 rad/s² | 1.71219 |
| `accelerations` | 4900 | .085872 | .38512 rad/s² | 1.31542 |

**Обновления оптимизатора не являются физическими шагами**; это явно указано
в metadata. Дополнительно в процессе отбора выполнены 33 полные оценки ×
6 эпизодов, **39600 физических управляющих шагов**, без физических аварий.

После обучения замороженные лучшие actor независимо перепроверены на 4 s
из 20°, 4 s из 21° и 20 s из 20°. Обе ветви проходят короткие условия 6/6.
На 20 s `actions` проходит **0/6**, конечные углы 24.70–26.87°, tail RMSE
3.09–5.15°; `accelerations` — **6/6**, 27.87–28.54°, tail RMSE 1.46–2.09°.
Эвристика на 20 s — 6/6, RMSE около .001°. Всего эта отдельная проверка:
**22800 физических управляющих шагов**, без обновления весов и без
эвристической помощи actor. Малая проверочная ошибка не равна качеству
управления. Ветка ускорений допускает большую совместную активацию
антагонистов; это не подтверждение экономичности или биологических синергий.

Inline-проверка артефактов подтвердила точное совпадение начальных
`policy.state_dict()` обеих ветвей с источником; неизменность критика и
нормализатора у best/last; непересечение episode IDs validation со всеми
учебными наборами; строгую загрузку всех best/last. Обычный `load_controller`
обе новые физики отвергает. ZIP-хеши после проверки не изменились.
Основной dynamics SHA-256 до/после:
`bc7cacd13f9bbe8dc777919fdd018a3e96a12ced614b3850f996c18af953e8e3`.
Исходник supervised runner, сохранённый в fingerprint:
`63c4f933a0633fcc1047b2a03335b3f18f29a0517a5948e6f6c837bdabb33b80`.

Отчёты и графики: `runs/d1_supervised_actuator/summary.json`,
`independent_evaluation.json`, `final_artifact_check.json`, `comparison.png`
и `comparison.svg`. Лучшие веса — `actions/best.zip` и
`accelerations/best.zip` с `.zip.metadata.json` в том же каталоге.
Данные/веса остаются игнорируемыми. Визуальный GUI-прогон в этом опыте
не выполнялся. Другие ступени, release и исходная задержанная/упругая
механика этими результатами не проверены; основной demo controller сохранён.

Дополнительный inline Python разбор без физических шагов и без обучения:
оба `best.zip` применены к прежнему `validation.npz`, через
`acceleration_tensor` вычислены ошибки и разложены по мышечным каналам.
Сумма вкладов совпала с точным ответом ядра с ошибкой <1e−4 rad/s².
На последней секунде teacher-эпизодов у `actions` ошибка разгибателей
около .00333 даёт −.743 rad/s², сгибателей +.259 rad/s²; суммарное смещение
−.484 rad/s², RMS .491 rad/s². У `accelerations` смещение −.0536 rad/s²,
RMS .0575 rad/s². Сохранены SHA исходных данных и checkpoint,
чувствительности и параметры Beta в
`runs/d1_supervised_actuator/loss_geometry_audit.json`. Это разбор ошибки
на одинаковых teacher-состояниях, не измерение постоянного ускорения в
замкнутом удержании и не новая проверка 20 s. Исходники физики/обучения и
веса не менялись; pytest повторно не запускался. `git diff --check` — exit 0.

## Абляция задержки, активации и сухожилия — 2026-11-04

По явному согласованию с владельцем добавлена отдельная диагностическая
физика `myo.actuator_ablation`: команда сразу равна активации, транспортная
очередь пуста, натянутое сухожилие нерастяжимо. Сохраняются геометрия MyoArm,
мышечные force–length/force–velocity, демпфирование, capacity law, контакт,
масса, timestep, reward, PPO/GAE, размер сети, начальные веса и нормализатор.
Неактивные каналы очереди явно обозначены и заполнены нулями; всего 773 входа.
Основные файлы физики/контроллера и предыдущие checkpoints не изменены.

Выполнены:

```bash
.venv/bin/python3 -m pytest -q tests/test_actuator_ablation.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo.train_actuator_ablation --mode instant_rigid \
  --output-dir runs/d1_actuator_instant_rigid \
  > runs/d1_actuator_instant_rigid.log 2>&1
.venv/bin/python3 -m myo.train_actuator_ablation --mode full --max-rollouts 2 \
  --output-dir runs/d1_actuator_full_check \
  > runs/d1_actuator_full_check.log 2>&1
.venv/bin/python3 scripts/compare_actuator_ablation.py
.venv/bin/python3 -m myo.train_actuator_ablation --mode instant_rigid \
  --resume runs/d1_actuator_instant_rigid/best.zip \
  --output-dir runs/d1_actuator_instant_rigid_continued \
  > runs/d1_actuator_instant_rigid_continued.log 2>&1
.venv/bin/python3 scripts/compare_actuator_ablation.py \
  --simple runs/d1_actuator_instant_rigid_continued \
  --output runs/d1_actuator_ablation_continued_comparison.json
.venv/bin/python3 -m py_compile src/myo/actuator_ablation.py src/myo/train_actuator_ablation.py tests/test_actuator_ablation.py scripts/compare_actuator_ablation.py
git diff --check
```

Три новых поведенческих проверки: **3 passed in 2.12 s**. Полный pytest
в sandbox: **88 passed, 1 failed** из-за запрета открыть UDP socket в прежнем
loopback-тесте. Повтор той же команды с доступом к локальному сокету:
**89 passed, 2 warnings in 19.95 s**, exit 0. Предупреждения SB3 прежние.
Компиляция и `git diff --check` — exit 0.

Физическая выполнимость отдельно проверена inline Python через
`evaluate_movement`: zero/heuristic × full/instant_rigid, **24 полных эпизода /
4800 управляющих шагов**. Эвристика прошла 6/6 в обеих физических моделях,
конечные углы около 29.99°/29.97°; zero — 0/6, около 17.17°/17.26°.
Отчёт `runs/d1_actuator_ablation_feasibility.json`. Ни подготовки учебного
старта, ни мышечных меток эти прогоны не предоставляли.

Контроль полного привода: exit 0, **2944 новых учебных шага**, первый блок
сравнения. Inline-проверка всех сохранённых массивов двух rollout и
`policy.state_dict()` после PPO установила совпадение с прежним первым блоком
`runs/d1_movement_curriculum` без численного расхождения. Отчёт
`runs/d1_actuator_full_equivalence.json`. Таким образом, новый runner не изменил
траектории и обновление полной контрольной модели на проверенном блоке.

Первый упрощённый запуск: exit 0, **47104 учебных шага**, остановка
`stage_budget_exhausted`. Лучший checkpoint на шаге 35328: score **8.74464**,
конечные углы **20.78–21.95°**, 0/6 успешных условий. У полного контроля
при том же бюджете лучший score **9.68471**, углы **19.59–21.44°**, также 0/6.
В реальных учебных пробах упрощённой модели максимум **144.68°**,
самое долгое попадание в допуски угла/скорости **.14 s**; у полной .12 s.
Отчёт равных первых бюджетов — `runs/d1_actuator_ablation_comparison.json`.
При исчерпании бюджета ещё не было шести блоков плато, поэтому выполнено
продолжение из лучших весов. Оно завершилось exit 0 по `stage_plateau` после
**17664 новых учебных шагов**: шесть блоков не улучшили score 8.74464.
Первый процесс занял 366.87 s, продолжение 137.06 s; вместе около 8 min 24 s.
Всего **64768 фактических учебных шагов** упрощённой модели, как у полного
предыдущего опыта с продолжением. Конечный счётчик ветки 52992: восстановлен
лучший checkpoint 35328, а не последний 47104.

За оба упрощённых запуска в допуск угла/скорости попали **58 отсчётов**, максимум
подряд по-прежнему .14 s; максимальный угол 144.87°. Ни одного физического
termination, успешных детерминированных условий 0/6. Продолжение полного
контроля также не улучшило его лучший результат. Это отрицательный результат
гипотезы «одного устранения этих трёх процессов достаточно для освоения
удержания»: ранний прогресс быстрее, но полезный навык не получен. Он не
доказывает, что задержки/упругость не влияют или GAE является единственной
причиной. Награда за позу и инерция тела сохраняют временную задачу и при
мгновенной активации. Отчёт второй ветки —
`runs/d1_actuator_ablation_continued_comparison.json`.

В новых NPZ сохраняются отдельные конечные наблюдения эпизодов. Независимая
inline-проверка первого rollout восстановила все семь timeout-bootstrap из
исходной сети и этих наблюдений: максимальная ошибка buffer reward
`2.234e-9`, конечный rollout value совпал точно. Оба экспериментальных initial
checkpoints загружаются через `load_ablation_controller` и отвергаются обычным
`load_controller` из-за отдельного physical fingerprint. Отчёт
`runs/d1_actuator_ablation_bootstrap_check.json`.

Финальная проверка четырёх экспериментальных checkpoints (лучшие обеих
упрощённых веток, последний block 6 и полный контрольный block 1) подтвердила
загрузку с правильной физикой и отказ обычного runtime. SHA всех 44 новых
упрощённых rollout совпали с отчётами; initial policy и нормализатор точно
совпадают с необученным источником. Запись —
`runs/d1_actuator_ablation_final_check.json`. Новые веса/данные остаются в
игнорируемом `runs/`, текущий demo controller не заменён.

Базовый физический SHA остаётся
`bc7cacd13f9bbe8dc777919fdd018a3e96a12ced614b3850f996c18af953e8e3`.
Упрощённая физика имеет отдельный fingerprint, включая actuator source и
точное описание режима. Это один фиксированный seed и пакет из трёх
упрощений; результат не разделяет их вклады и не устанавливает анатомическую
достоверность или готовность нового демо. Программа —
[movement-curriculum.md](movement-curriculum.md#диагностическое-сравнение-приводов--2026-11-04).

## Разбор фактического поэтапного обучения — 2026-11-03

После запроса владельца разобраны исходные сохранённые буферы и checkpoints,
без `model.learn`, optimizer updates или новых весов. Выполнены:

```bash
.venv/bin/python3 scripts/audit_movement_training.py \
  > runs/d1_movement_training_audit.log 2>&1
.venv/bin/python3 scripts/audit_movement_replay.py \
  > runs/d1_movement_replay_audit.log 2>&1
.venv/bin/python3 -m py_compile scripts/audit_movement_training.py scripts/audit_movement_replay.py
git diff --check
```

Команды завершились с кодом 0. Первый скрипт прочитал **44 настоящих rollout /
64768 учебных шагов**, измерил фазовые награды/advantage, изменение log-prob
после 22 блоков, веса/латенты/разброс восьми checkpoints на общем наборе
состояний. Пересчёт GAE в float64 на всех буферах: максимальная ошибка
`1.093e-6` в учебном масштабе, в пределах float32. Сверены returns, флаги
границ эпизодов, timeout и конечный rollout bootstrap. Отдельные terminal
observations auto-reset не записаны, поэтому этот аудит не пересчитывает
внутренние timeout-bootstrap независимо от сохранённых buffer rewards.
Нормализованный знак advantage иллюстрируется 64 новыми перестановками
minibatches; исторические PPO-перестановки не восстановлены. Второй запуск
этого скрипта после уточнения float64-пересчёта не делал физических шагов.

Второй скрипт выполнил **2272 новых физических управляющих шага**:
400 шагов полного replay двух эпизодов, 672 шага повторения их префиксов
и 1200 шагов двенадцати двухсекундных продолжений. Сеть заморожена.
Все 400 исходных наблюдений совпали без численного расхождения; награды
согласованы до `2e-6`. Источник — настоящий rollout до обновления block 13;
восстановленные old log-prob отличаются максимум на `1.431e-5`.
Шесть ветвей каждого состояния имеют одинаковый hash начального наблюдения
и проверенные префиксы. Импульсы проходят через обычные мышечные команды,
задержку, активацию и сухожилия, без подмены torque или payload.

Из 25.47°/+35.19°/s средняя команда сети даёт перелёт к 143.57° за 2 s.
Ослабление сгибателей на .05 в течение 200 ms даёт через .5 s 30.97°/+4.16°/s,
но после возврата сети рука снова разгоняется. Из 29.91°/+1.50°/s средняя
команда сети опускает руку к 19.43°, тогда как эвристика удерживает 30.01°.
Накопленный GAE в рассмотренных ветвях предпочитает физические улучшения;
одношаговый TD иногда имеет противоположный знак. Продолжения детерминированы,
а V относится к стохастической policy: это физические сравнения, не точный Q.
Ограниченные 2-секундные возвраты не включают последующие последствия.

Отчёты с source hashes: `runs/d1_movement_training_audit.json` и
`runs/d1_movement_replay_audit.json`. Чтение/проверки исходных ZIP, сохранённых
нормализаторов и physical fingerprint подтвердили неизменность исходных
артефактов. Подробная интерпретация и границы выводов —
[controller-learning-review.md](controller-learning-review.md#поэтапное-движение-фактические-данные).
Полный pytest не повторялся: в этом разборе изменены только диагностические
скрипты и документация, core dynamics/reward/observations/serialization прежние.

## Поэтапное обучение движения — 2026-11-03

Реализованы фиксированные ступени `myo.movement_curriculum` и runner
`myo.train_movement`: переход после двух полных успешных проверок, сохранение
лучшего результата, остановка по плато/бюджету, фактические rollout и строгая
проверка совместимости продолжения. Физика, direct-Beta action mapping,
формула reward и PPO/GAE не менялись. Программа и результаты —
[movement-curriculum.md](movement-curriculum.md).

Выполнены команды:

```bash
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo.train_movement --output-dir runs/d1_movement_curriculum \
  > runs/d1_movement_curriculum.log 2>&1
.venv/bin/python3 -m myo.train_movement \
  --resume runs/d1_movement_curriculum/stage_0_best.zip \
  --output-dir runs/d1_movement_curriculum_continued \
  > runs/d1_movement_curriculum_continued.log 2>&1
.venv/bin/python3 -m py_compile src/myo/movement_curriculum.py src/myo/train_movement.py tests/test_movement_curriculum.py tests/test_train_movement.py
git diff --check
```

Первый pytest в sandbox упёрся в `PermissionError` существующего теста loopback
UDP. Повтор полного pytest вне ограничения sandbox: **86 passed, 2 warnings
in 15.56 s**, exit 0. Это включает три проверки новых сценариев/runner:
холодное причинное начало, непрерывность физики при смене команды, отказ gates
при пассивной опоре/пролёте/неполном окне и совместимость resume. Два прежних
предупреждения SB3 относятся к нормализованному Box. Компиляция и
`git diff --check` также завершились с кодом 0.

Первый учебный процесс: exit 0, **47104 шага / 32 rollout / 457.89 s**,
`stage_budget_exhausted`. Лучшая оценка первого этапа улучшилась с 14.86343
до 9.68471 на шаге 41216, но успешных условий оставалось 0/6. На конце бюджета
были только три блока без значимого улучшения, поэтому выполнено продолжение
из лучших весов. Оно завершилось exit 0, **17664 новых шага / 12 rollout /
173.80 s**, `stage_plateau`: шесть блоков не улучшили исходную лучшую оценку.
Суммарно выполнено **64768** учебных шагов; конечный счётчик ветки — 58880,
поскольку продолжение восстановило block 14, а не последний block 16.
Этапы 1–5 не запускались. Успешной новой модели для демо нет.

В лучшей проверке без шума конечные углы 19.59–21.44° вместо цели 30°.
Отдельным вызовом `evaluate_movement` из inline Python проверены zero и
`baseline_action`: по шесть полных 4-секундных эпизодов, **2400 физических
управляющих шагов суммарно**, без optimizer и подготовки учебных стартов.
Эвристика прошла 6/6 (конечный угол около 29.99°), zero 0/6 (17.17°).
Отчёт `runs/d1_movement_curriculum_feasibility.json`.

Два разбора NPZ настоящего обучения, без новых симуляций, подтвердили посещение
углов выше пассивной поддержки и большие перелёты. Первый запуск: максимум
144.987°, выше 25° на 4.908% шагов; продолжение: максимум 145.037°, выше 25°
на 7.841%. В допуски угла ±3° и скорости ≤5°/s попали 32 и 13 отсчётов
соответственно. Самые длинные непрерывные серии — 0.12 s и 0.06 s; границы
эпизодов учтены. Отчёты `runs/d1_movement_curriculum_discovery.json` и
`runs/d1_movement_curriculum_continued_discovery.json`. Это обнаруженные краткие
состояния, а не успешное удержание и не доказательство конкретной причины плато.

Независимый пересчёт первого сохранённого rollout подтвердил GAE и returns
с максимальными расхождениями `6.48e-8` и `6.41e-8`; все массивы конечны.
Конечный bootstrap проверен по сохранённому наблюдению и исходной сети.
Ограничение: отдельные terminal observations при auto-reset не записаны,
поэтому внутренние timeout-bootstrap нельзя независимо пересчитать только
из NPZ. EV в отчётах относится к bootstrapped targets и сама по себе не
устанавливает верное ранжирование действий.

Финальная проверка через `load_controller` и `PPO.load` загрузила лучшие
checkpoints обоих запусков и последний block 6 продолжения. Совпали ZIP SHA,
счётчики, curriculum descriptor и fingerprints исходников runner, политики,
наблюдений и физики. Запись —
`runs/d1_movement_curriculum_final_artifact_check.json`; новых физических шагов
или обновлений весов эта проверка не делала. Физический SHA остался прежним:
`bc7cacd13f9bbe8dc777919fdd018a3e96a12ced614b3850f996c18af953e8e3`.
Новые веса и учебные данные сохранены только в игнорируемом `runs/`.

## Копии actor: одно ограниченное обновление GAE/MC — 2026-11-03

После согласия пользователя на короткий опыт выполнены:

```bash
.venv/bin/python3 scripts/audit_mc_actor_update.py
.venv/bin/python3 -m py_compile scripts/audit_mc_actor_update.py
git diff --check
```

Все команды завершились с кодом 0. Основной запуск занял 62.81 s:
3 учебных и 12 проверочных 10-секундных эпизодов, **7500 управляющих шагов**.
Условия — исходное упражнение 20→75°, 1 kg, trigger/capacity=1; small block 3
выбран до сбора данных. Учебные seeds 0/1/2, проверочные 10/11/12 плюс
детерминированный прогон исходной и двух изменённых копий.

В памяти выполнены реальные actor-only optimizer steps, critic заморожен.
GAE(.95) и `MC_return - V_old` используют одинаковые свежие данные, исходные
веса/Adam и порядок minibatches. В обеих ветках **16 обновлений из 24**:
KL early stop перед 17-м. Это не завершённая эпоха и не длительное обучение.
Косинус градиентов .819; знак нормализованного advantage различается на 29.3%
отсчётов. Средний проверочный scaled discounted return: исходная −1.71105,
GAE −1.70787, MC −1.71454. Удержаний цели нет; deterministic около 15°.

Ограничения: конец 10 s объявлен настоящим терминальным концом диагностического
задания, с `V_terminal=0`, а исходный critic обучался с timeout-bootstrap.
Свежие данные содержат лишь микроподъёмы 1.19–2.72° у упора, не полноценное
полезное движение. Сохранены minibatch-нормировка и старый импульс Adam;
clipping применяется только к actor. Три проверочных seed не устанавливают
превосходство метода или способность обучиться на успешных подъёмах.

Отчёт `runs/d1_mc_actor_update_audit.json`, свежие данные в соседнем NPZ.
Повторным независимым пересчётом проверены все 1500 MC-return/advantage,
число проверочных эпизодов, SHA скрипта, исходных ZIP/metadata и физики.
Физический SHA прежний:
`bc7cacd13f9bbe8dc777919fdd018a3e96a12ced614b3850f996c18af953e8e3`.
Исходные веса, critic и буферы неизменны; новые weight-файлы не сохранялись.
Полный pytest не перезапускался: core dynamics/reward/serialization не менялись.
Разбор опыта и источники по будущей иерархии —
[controller-learning-review.md](controller-learning-review.md).

## Нагрузка и обнаружение полезных действий — 2026-11-03

Выполнена дополнительная диагностика без обучения, шагов optimizer, изменения
physics/reward, конфигураций и checkpoints. Подробный разбор и ограничения —
[controller-learning-review.md](controller-learning-review.md).

Фактические команды:

```bash
.venv/bin/python3 scripts/audit_rest_reward.py
.venv/bin/python3 scripts/audit_excitation_barrier.py
.venv/bin/python3 scripts/audit_exploration_effort.py
.venv/bin/python3 -m py_compile scripts/audit_rest_reward.py scripts/audit_excitation_barrier.py scripts/audit_exploration_effort.py
git diff --check
```

Первый процесс остановлен Ctrl-C, **exit 130**, после сохранения двух полных
наборов zero/heuristic. Его первоначальный default включал ещё два контроллера;
ни один дополнительный NN-эпизод не выполнен. В окончательном скрипте default
содержит только zero/heuristic, повторных физических прогонов не было.
Оба набора завершили по 18 упражнений / 14700 шагов: **29400 шагов суммарно**.
Суммы проверены по сохранённым NPZ; использована компактная награда на каждом
шаге, не исходные tracking-компоненты базовой среды. Средняя raw-награда по
времени −1.66033 у zero и +0.31963 у heuristic; последняя лучше во всех 18
парных эпизодах также по конечному дисконтированному возврату с gamma=0.99.

Импульсный аудит завершился успешно: **54 импульса + 11 контролей**, каждый
с 1 s пассивной подготовки и 2 s наблюдения; **9750 управляющих шагов**,
124.6 s wall time. Все 65 проб прошли полностью без физических/численных
срывов. Внутри каждой массы совпадают хеши начального состояния, мышц, очереди
и истории. Массы 0.25/1/2 kg задавались при reset отдельных диагностических
эпизодов. Для 1 kg команда сгибателям 0.3 на 500 ms дала максимум 81.68° при
нулевых разгибателях и 17.94° при команде разгибателям 0.1. Даже импульс 0.1
на 20 ms дал небольшое улучшение угла и фактической награды относительно покоя.
Эти примеры не устанавливают вероятность случайного открытия или навык удержания.

Аудит усилий использовал только пять сохранённых frozen-probes, без новой
физики. На каждом рассмотрено 2300 отсчётов после первых 0.5 s и при цели >15°.
Сверены хеши NPZ/checkpoints и порядок наблюдений; восстановленные средние
policy совпали с записанными до `9.84e-7`. Строгие пороги одновременных команд
не объявляются необходимыми условиями движения; команды, активации и полное
натяжение сухожилия анализируются раздельно.

Отчёты: `runs/d1_rest_reward_audit.json` (и два NPZ),
`runs/d1_excitation_barrier_audit.json`, `runs/d1_exploration_effort_audit.json`.
Компиляция трёх скриптов и `git diff --check` — код 0. Подтверждены прежний
физический SHA `bc7cacd13f9bbe8dc777919fdd018a3e96a12ced614b3850f996c18af953e8e3`
и SHA всех initial/four-block checkpoints обеих сетей. Полный pytest не
перезапускался: изменения касаются диагностических скриптов и документации.

## GAE и случайно найденные улучшения — 2026-11-03

Продолжена диагностика без обучения, изменения reward/физики и шагов optimizer.
Успешно выполнены:

```bash
.venv/bin/python3 scripts/audit_training_records.py --replay-first
.venv/bin/python3 scripts/audit_gae_targets.py
.venv/bin/python3 scripts/audit_exploration_discovery.py
.venv/bin/python3 -m py_compile scripts/audit_training_records.py scripts/audit_gae_targets.py scripts/audit_exploration_discovery.py
.venv/bin/python3 -m pytest -q tests/test_compact_protocol.py::test_fixed_protocol_cycles_conditions_and_preserves_physics_across_commands
git diff --check
```

Отчёты: `runs/d1_training_records_audit.json`, `runs/d1_gae_target_audit.json`,
`runs/d1_exploration_discovery_audit.json`; пошаговые массивы сохранены рядом
в NPZ. Компиляция трёх скриптов и `git diff --check` — код 0. Полный pytest
повторно не запускался: проверены диагностические скрипты с замороженными весами.
Существующий тест соответствия команд и награды — **1 passed in 2.23 s**.
Итоговая проверка SHA подтвердила неизменность физики и всех 10 checkpoints;
хеши двух скриптов с записанным `script_sha256` совпадают с их отчётами.

Из пересекающихся Monitor-буферов checkpoints восстановлены **144 исходные
суммарные награды завершённых эпизодов для каждой архитектуры**: 117600 шагов
и ещё 160 шагов незавершённого эпизода дают 117760. Это подлинные агрегаты
обучения, но не пошаговые траектории. Первый сбор отдельно воспроизведён из
initial-checkpoint: по 1472 шага, **2944 суммарно**, без `model.learn`;
все сохранённые статистики первого rollout совпали. Реконструкция не выдаётся
за найденную запись исходной траектории и не восстанавливает поздние обновления.

Проверка реального SB3 collector на шести синтетических переходах подтвердила
обработку termination, truncation и границы buffer: максимальная ошибка GAE
`7.90e-8`. В каждом реконструированном первом rollout независимо проверены
1471 внутренних равенства GAE, максимальная ошибка `3.32e-8`. Единственная
поправка к награде для timeout находится на шаге 1449. Последнее равенство
буфера отдельно не проверено: `last_value` реконструкции не сохранялся;
этот путь покрыт синтетической проверкой collector.
На тех же ранее сохранённых контрфактических траекториях сравнили
lambda 0.9/0.95/0.99/1.0. При горизонте 2 s первые три значения сохраняют
неверный знак сравнения во всех шести парах; lambda=1 совпадает с направлением
реальной награды в 6/6. При 0.5 s даже lambda=1 ошибается в 5/6 из-за конечного
bootstrap. Это пересчёт оценок, **не результат обучения с новой lambda**;
равенство GAE(1) возврату с bootstrap является алгебраическим тождеством.

Исследование случайных действий повторно использовало четыре набора
initial/final из предыдущей диагностики и собрало только шесть промежуточных
checkpoints: **14700 новых физических шагов**, один seed, по три номинальных
упражнения. В сумме проверены 24500 шагов десяти замороженных policy.
В 23000 отсчётах после исключения первых 0.5 s reset и цели 15° нет положительной
raw-награды или устойчивого удержания 0.3 s. При этом менее отрицательная
награда и частичный прогресс встречаются: small block 3 при неизменной цели
125° поднимается **14.873→21.686° за 0.26 s**, raw-награда меняется
**−3.825→−3.476**; превышение 20° длится 0.1 s. В финальном small есть подъём
14.641→21.027° за 0.42 s, но он тоже не закрепляется в управляемом движении.

В окне последнего подъёма средняя raw-GAE положительна (+0.04425), а после
центрирования по всему диагностическому набору отрицательна (−0.9372).
Это **не фактическая нормировка PPO minibatch**. Вероятности тех же сохранённых
действий по разным checkpoints меняются в обе стороны: окно block 3 становится
менее вероятным в block 4, собственное полезное окно block 4 — более вероятным.
Следовательно, утверждение «все найденные улучшения подавляются» не подтверждено.
Сравнение вероятностей на фиксированных наблюдениях не доказывает, что именно
эти действия встретились при обучении или вызвали конкретное обновление весов.
Подробная интерпретация — [controller-learning-review.md](controller-learning-review.md).

## Диагностика без продолжения обучения — 2026-11-03

По просьбе владельца обучение не возобновлялось: выполнены чтение checkpoints,
замеры представлений и короткие физические пробы с замороженными весами.
Ни `model.learn`, ни шаги optimizer не вызывались; физика, reward и смысл
прямых возбуждений сохранены. Разбор и сопоставление с MuscleMimic —
[controller-learning-review.md](controller-learning-review.md).

Успешно выполнены:

```bash
.venv/bin/python3 scripts/audit_policy_representations.py
.venv/bin/python3 scripts/audit_policy_behavior.py --output runs/d1_architecture_behavior_audit.json
.venv/bin/python3 scripts/audit_critic_signal.py
.venv/bin/python3 scripts/audit_critic_counterfactual.py --output runs/d1_critic_counterfactual_audit.json
.venv/bin/python3 scripts/audit_stochastic_counterfactual.py --output runs/d1_critic_stochastic_counterfactual_audit.json
.venv/bin/python3 -m py_compile scripts/audit_policy_representations.py scripts/audit_policy_behavior.py scripts/audit_critic_signal.py scripts/audit_critic_counterfactual.py scripts/audit_stochastic_counterfactual.py
git diff --check
```

Компиляция пяти скриптов и проверка пробелов завершились с кодом 0. Отчёты
представлений и критика: `runs/d1_architecture_weight_audit.json` и
`runs/d1_critic_signal_audit.json`; остальные указаны в командах. Это
диагностика поведения, не новый запуск pytest или свидетельство успешного демо.
Финальная проверка также подтвердила неизменность физического SHA и всех
10 ZIP (initial + четыре блока каждой архитектуры). Контрольные суммы отчётов
сохранены в `runs/d1_learning_review_manifest.json`.

- Обе последние policy теряют статические цели 20°/45°/75°/120°, в том числе
  после 2 s реального удержания эвристикой. В 23 коротких пробах выявлена
  поддержка руки нижним ограничением при продолжающемся возбуждении мышц;
  причина короткого провала не в утомлении. При цели 75° small создаёт около
  −8.72 Nm мышечного момента, wide — −1.02 Nm, то есть разгибает руку.
  Переключение на эвристику из точно той же истории возвращает угол к
  71.33°/71.66° за 2 s без reset и изменения нагрузки.
- Веса меняются; расширение сети само по себе не устранило поведенческую
  ловушку. Общая explained variance критика относительно GAE-целей около
  0.98 скрывает показатели подъёма/опускания: **0.051/−0.790** у small и
  **−0.052/−0.603** у wide. Отрицательная explained variance при малом
  разбросе целей сама по себе не доказывает причину провала.
- Выполнены 22 сравнения действий из одинаковых физических историй с
  deterministic-продолжением, затем отдельные 9 ветвей wide из одного
  стохастически достигнутого состояния. В последней проверке использованы
  три общих массива квантилей Beta: при одиночном `flexors +0.05` фактическая
  дисконтированная награда за 2 s улучшилась во всех трёх продолжениях, а GAE
  уменьшилась; обратное вмешательство ухудшило награду, но повысило GAE.
  Это локальная ошибка ранжирования при стохастическом продолжении, не
  доказательство единственной причины неудачи. Эффект действий мал, рука
  остаётся около нижнего упора; три пробы не являются проверкой обобщения.

Для общих случайных квантилей использована имеющаяся `mpmath`: максимальный
остаток Beta CDF `1.04e-10`; контроль известных квантилей дал ошибку не более
`5.7e-12`. Новые зависимости не устанавливались. Попытка
`.venv/bin/python3 runs/plot_architecture_comparison.py` завершилась отсутствием
`matplotlib`; график не создан, анализ выполнен по числовым отчётам.

## Сравнение малой и расширенной MLP — 2026-11-02

Запущены два новых локальных CPU-опыта с seed 0: прямые Beta-команды,
`--hidden-layers 128 128` и `--hidden-layers 1024 1024 512`, каталоги
`runs/d1_architecture_small_tanh` / `runs/d1_architecture_wide_tanh`.
Команды и условия — в [bounded-controller.md](bounded-controller.md).
Tanh, reward, физика, PPO, статистики наблюдений и 18 сценариев одинаковы;
не добавлены LayerNorm, память команды или teacher-метки. Физический SHA прежний.
Изменение ширины/глубины проверяется совместно, на одном seed; вывод об общем
превосходстве архитектуры из этого сравнения не следует.

Начальная проверка `runs/d1_architecture_initial_comparison.json`:

| Policy | Обучаемых параметров, actor+critic | Среднее время deterministic-вызова |
| --- | ---: | ---: |
| 128×128 | 232845 | 0.211 ms |
| 1024×1024×512 | 4740621 | 0.826 ms |

По 500 вызовов на CPU/один поток при параллельно работающем обучении;
это локальная оценка стоимости forward, не гарантия задержки GUI.
Runtime-загрузка обоих начальных ZIP побитово сохраняет deterministic-действия.
Центрирование входов совпало; все тензоры малой начальной policy совпали
с `runs/d1_bounded_beta/initial.zip`. Метаданные сохраняют фактическую архитектуру,
runner SHA и прежние физические/action/history контракты. Снимок исходника runner
сохранён в обоих каталогах как `runner.py`.
Отдельная попытка продолжить широкий checkpoint с `--hidden-layers 128 128`
завершилась ожидаемым `ValueError` до обучения; отчёт —
`runs/d1_architecture_resume_guard.json`. `git diff --check` — без ошибок.

После добавления выбора архитектуры выполнено `.venv/bin/python3 -m pytest -q`:
**82 passed, 1 failed, 2 warnings in 19.29 s**. Единственный отказ — запрет
UDP-сокета в sandbox. Отдельный запуск
`.venv/bin/python3 -m pytest -q tests/test_telemetry.py::test_command_receiver_ignores_old_episode_and_timeout_only_requests_relaxation`
с разрешённым loopback: **1 passed in 0.21 s**. Все 83 проверки прошли
в двух запусках; это не результат обучения.

Обе серии завершились по `plateau` на **117760 шагах**. Последние оценки:
**43.12104** для 128×128 и **43.16332** для 1024×1024×512; у обеих **0/12
движений и 0/6 проходов**. На 2026-11-03 обучение не выполняется, полезная
новая demo-policy не получена. Подробности диагностики —
[controller-learning-review.md](controller-learning-review.md).

## Отказ от интегрирования выхода сети — 2026-11-02

Предложение трактовать нулевой выход как сохранение предыдущего возбуждения
отклонено: оно вводит искусственную память команды без биологического
обоснования. Удалены residual-policy, её initializer, runtime-интеграция и тест
этого отвергнутого контракта. Исходники сохранены только в ignored
`runs/rejected_residual_design/` для происхождения экспериментальных артефактов.
Были выполнены только инициализация и проверка необученных весов, `trained_steps=0`;
обучение PPO на этой архитектуре не запускалось. Успех короткого подготовленного
удержания не является выученным навыком.

В рекомендуемом контракте действие — прямое возбуждение 0…1. Нулевая команда
проходит обычную моторную задержку и динамику активации; история остаётся
наблюдением. Физические состояния и их инерция не заменяются интегратором policy.
Предшествующая persistent Beta также добавляет искусственную динамику;
её исходник/загрузка сохранены для воспроизводимости, рекомендация убрана.

Итог уже завершённых серий (оценка меньше — лучше):

| Серия | Статус / actor-шаги | Лучшая оценка | Успешные задания |
| --- | --- | ---: | --- |
| Persistent Beta, 10 эпох | plateau / 264960 | 37.343 | 0/12 движений, 0/6 проходов |
| Persistent Beta, 1 эпоха | plateau / 117760 | 42.600 | 0/12 движений, 0/6 проходов |
| Подготовленное удержание | stage_plateau / 19136 | 9.444 | 3/18 коротких удержаний в лучшем блоке |

Лучший учебный блок удержания (13248 шагов) отдельно проверен на номинальных
холодных командах: подъём 20→75° завершился на 15.00°, опускание 75→20° —
на 15.49°, устойчивого достижения нет. Отчёт:
`runs/d1_prepared_holding/best_stage_nominal.json`.
Новая демонстрируемая policy не получена; эти серии больше не выполняются.

После удаления отвергнутой архитектуры выполнено:

```bash
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m pytest -q tests/test_telemetry.py::test_command_receiver_ignores_old_episode_and_timeout_only_requests_relaxation
git diff --check
```

Полный запуск в sandbox: **82 passed, 1 failed, 2 warnings in 15.52 s**;
единственный отказ — `PermissionError` при создании локального UDP-сокета.
Этот же тест отдельно с разрешённым loopback: **1 passed in 0.16 s**.
Таким образом, все 83 проверки прошли в двух запусках, без изменения теста
обмена командами. `git diff --check` — без ошибок. Это проверка удаления
архитектуры и совместимости оставшихся путей, не успех обучения.

## Подготовленные состояния для обучения удержанию — 2026-11-02

Добавлены `PreparedProtocol` и `myo.train_holding`. Перед actor-эпизодом
эвристика 2 s управляет обычными мышечными командами; затем actor получает
реальное текущее состояние с сохранением fatigue/задержки/истории/времени.
В buffer идут только действия actor; teacher-меток нет. Физический SHA прежний.
Этапы и ограничения — в [prepared-curriculum.md](prepared-curriculum.md).

Выполнена физическая проба холодных/подготовленных состояний для 45°/75°/120°,
начальной и обученной persistent Beta, seed 0. В холодных двухсекундных пробах
средний settled bonus практически нулевой; после подготовки 0.025–0.074.
Это доступность полезных состояний, не успешное удержание. Отчёт и снимок скрипта
сохранены в `runs/d1_prepared_holding_probe.json`, `runs/probe_prepared_holding.py`.

```bash
.venv/bin/python3 -m myo.train_holding --output-dir runs/d1_prepared_holding
.venv/bin/python3 -m pytest -q tests/test_compact_protocol.py
.venv/bin/python3 -m pytest -q
```

Целевой набор: **8 passed in 6.73 s**. Полный: **83 passed, 2 warnings in
37.77 s**, с разрешённым UDP loopback. Новая регрессия сравнивает handoff с
непрерывным прогоном, сохраняет изменившуюся capacity и фактическую историю,
проверяет порядок открытия команды и запрещает teacher при actor-шагах.
Предупреждения SB3 прежние. Наличие зелёных тестов не означает освоения навыка.

Первые четыре блока короткого удержания (1472/2944/4416/5888 actor-шагов)
дали учебные оценки **14.595/13.661/12.503/11.695**, успешных коротких
проверок **0/3/2/1 из 18**. Бонус >0.5 встречался примерно в 13–14% учебных
шагов. Это ещё полусекундные подготовленные упражнения; main cold-проверка
ими не заменяется. Это промежуточная запись; окончательный статус —
`stage_plateau` на 19136 actor-шагов (см. итог выше и `summary.json`).

Контроль с одной эпохой PPO завершился по **плато** на **117760 шагах**:
оценки 43.190/43.147/42.939/42.600, во всех блоках 0/12 движений и 0/6 проходов.
Уменьшение повторных обновлений само по себе не дало движения. Контроль с
десятью эпохами на 176640 шагах дал 37.551, также 0/12 и 0/6; номинальный
подъём заканчивается на 17.56°. Улучшение общей оценки не означает
освоения подъёма. Контроль затем завершился плато на 264960 шагах;
окончательные показатели приведены выше.

## Ограниченные возбуждения и авторегрессионное исследование — 2026-11-02

Добавлены `BetaExcitationPolicy` и `PersistentBetaPolicy`: пробы сразу лежат
в (0,1), deterministic-действие равно среднему распределения физических
возбуждений. У второго варианта среднее условной пробы зависит от предыдущей
команды в наблюдении; log-probability при replay использует ту же записанную
историю. Это часть policy, а не внешний фильтр или изменение динамики мышц.
Схема, параметры и команды — в [bounded-controller.md](bounded-controller.md).

Фактически запущены:

```bash
.venv/bin/python3 -m myo.train_bounded --output-dir runs/d1_bounded_beta
.venv/bin/python3 -m myo.train_bounded --persistence 0.9 \
  --output-dir runs/d1_bounded_persistent_beta
```

В обоих случаях обучение с нуля по физическим результатам, без teacher-меток,
CPU/один поток/seed 0, прежние 18 упражнений. Физический SHA остаётся
`bc7cacd13f9bbe8dc777919fdd018a3e96a12ced614b3850f996c18af953e8e3`.
Reward/value масштабированы на 0.01, входы центрированы по прежним обучающим
целым эпизодам, rollout 1472, minibatch 64, lr=0.0003, GAE lambda=0.95,
gamma=0.99, до 10 эпох, target KL=0.02. Исходная награда и критерии не менялись.

Независимая Beta после 29440/58880 шагов дала оценку **43.213/43.148**, оба раза
0/12 движений и 0/6 проходов. Серия остановлена для переработки исследования,
статус `stopped_for_exploration_redesign`, а не формальное плато. Последний
полный checkpoint — 58880; последний завершённый rollout до прерывания —
73600. Эти дополнительные несохранённые шаги не входят в выбранные веса.
Начальные номинальные подъём/опускание также проверены: оба закончились около 15°.

Физическая проба необученного авторегрессионного варианта на том же seed 0
посетила 99.40° в 10-секундном подъёме при persistence=0.9; при 0.97 максимум
59.06°. Это охват, не освоение команды. Отчёт — ignored
`runs/d1_persistent_beta_coverage.json`, использованный скрипт сохранён рядом.
Коэффициент 0.9 обеспечивает связь около 190 ms; он также меняет deterministic
динамику policy, поэтому сравнение не изолирует только случайный шум.

Targeted-команды `.venv/bin/python3 -m pytest -q tests/test_policy_beta.py`:
сначала **2 passed in 2.19 s**, после авторегрессионного варианта —
**3 passed in 3.59 s**. Полный набор после двух вариантов — **82 passed,
2 warnings in 46.60 s** с разрешённым UDP loopback. До этого независимая
версия дала **81 passed, 2 warnings in 33.26 s**; обычный sandbox блокировал
один UDP-тест (`80 passed`, `PermissionError` при создании сокета), поэтому
набор повторён с разрешённым локальным loopback, без изменения тестов.

Регрессии проверяют допустимость проб/совпадение среднего, сохранение
существенных возбуждений у расслабления, конечные likelihood/градиенты,
обновление PPO и перенос checkpoint. Для условного распределения дополнительно
проверены восстановление предыдущей команды из нормализованного наблюдения и
совпадение log-probability при перемешанном replay. Вручную проверены runtime
обеих policy, отказ при неверном/отсутствующем SHA и при подмене коэффициента
авторегрессии. Эти проверки не являются доказательством освоения движения.

После усиления runtime-проверки коэффициента/привязки истории полный набор
повторён: **82 passed, 2 warnings in 39.17 s**. `git diff --check` — без ошибок.

Дополнительное сравнение числа эпох:

```bash
.venv/bin/python3 -m myo.train_bounded --persistence 0.9 --epochs 1 \
  --output-dir runs/d1_bounded_persistent_one_epoch
```

Начальные 21 тензор двух авторегрессионных вариантов совпали побитово;
сравнение сохранено в `runs/d1_bounded_epoch_initial_comparison.json`.
У варианта с десятью эпохами после первого блока Adam выполнил 3433 шага,
у прежнего gSDE mixed на таком же физическом бюджете — 295. Поэтому отдельно
проверяется уменьшение повторного использования rollout; это гипотеза,
а не доказанная причина провала. Награда, seed, физика и критерии одинаковы.

На момент прерывания рабочей сессии варианты ещё выполнялись. У десяти эпох
полные проверки на 29440/58880/88320 шагах дали **42.787/42.921/42.870**,
все 0/12 движений и 0/6 проходов; последний наблюдавшийся rollout — 111872.
У одной эпохи первая полная проверка на 29440 дала **43.190**, также 0/12
и 0/6; собран блок до 58880, его оценка ещё выполнялась. Это промежуточные
результаты, не окончание серий и не полезный демонстрационный контроллер.
Итоговый статус/отчёты следует читать в `summary.json` соответствующих
каталогов: runner автоматически останавливается при готовности или плато.

## Локальная проверка удержания и распределения действий — 2026-11-02

Выполнен `scripts/probe_policy_holding.py` для
`runs/d1_feedback_refine/cycle_02.zip` и `runs/d1_feedback_mixed/cycle_02.zip`.
На каждом checkpoint проверены номинальные подъём/опускание по 10 s:
детерминированная policy, переключение на эвристику через 5 s, эвристика с начала.
Команды и физика сохранены, обучение не выполнялось. У первых двух вариантов
совпали хеши последовательностей наблюдений до переключения. Полные команды
и ограничения — в [ppo-diagnosis.md](ppo-diagnosis.md).

У refine переключение повысило среднюю награду последних 2.5 s
с −1.202 до +0.910 в подъёме и с −0.252 до +0.897 в опускании; конечные углы
74.94° и 19.84°. Это проверка доступной лучшей альтернативы, не имитационное
обучение. У нижнего ограничителя сырые средние двух сгибателей ≈−0.464/−0.440
при σ≈0.1: вероятность возбуждения >0.05 составляет ≈1.31e−7/4.98e−7 на
посещённых наблюдениях. Исследование после обрезания здесь сильно ограничено.
Эти маргинальные вероятности не означают независимых проб во времени.

Отчёты — ignored `runs/d1_feedback_holding_probe.json` и
`runs/d1_feedback_mixed_holding_probe.json`. Исходники dynamics, наблюдений,
reward, checkpoint/runtime и защищённой policy не изменялись; для добавленного
диагностического скрипта проверено фактическое выполнение. Предыдущий полный
набор из 79 тестов не повторялся. `git diff --check` — без ошибок.

## Диагностика PPO: шум, окно обновлений и удержание — 2026-11-02

Выполнены семь локальных CPU-серий `scripts/probe_ppo_feedback.py`, seed 0,
по 29440 новых управляющих шагов. Пять стартовали с общего начального actor/critic,
два продолжили один checkpoint с 29440 до 58880. Физика, груз 1 kg, задержки,
сухожилия, 18 фиксированных упражнений и критерии готовности сохранены.
Полные команды и ограничения — в [ppo-diagnosis.md](ppo-diagnosis.md).

| Опыт | Итоговая оценка (меньше лучше) | Движения / проходы |
| --- | ---: | --- |
| Обычный gSDE, 20 ms, rollout 128 | 43.380 | 0/12, 0/6 |
| Обычный gSDE, 200 ms, rollout 128 | 45.551 | 0/12, 0/6 |
| Unit-latent gSDE, 200 ms, rollout 128 | 40.394 | 0/12, 0/6 |
| Unit-latent gSDE, 200 ms, rollout 1472 | 27.550 | 0/12, 0/6 |
| Продолжение предыдущего с прежним шумом | 29.996 | 0/12, 0/6 |
| С нуля, rollout 1472, GAE lambda=1 | 38.961 | 0/12, 0/6 |
| Продолжение mixed с σ=0.1 | 23.980 | 0/12, 0/6 |

У обычного gSDE фактическая медиана σ выросла до ≈1.79/2.07. Нормирование
признаков только для шума удержало σ≈0.54; проверены выборка и log-probability.
Увеличение rollout улучшило RMSE номинального прохода ≈62→29°, но одновременно
уменьшило число Adam-обновлений 1142→295, поэтому механизм эффекта не изолирован.
У лучшего итогового варианта конец номинального подъёма 77.61° при цели 75°,
но RMSE последних 5 s 35.29°, установления нет. Опускания упираются примерно
в 14.9°. Ни один из этих checkpoint не признан готовым для демо.

Дополнительно выполнены физические пробы корреляции шума, реакции на разные
команды из общего префикса и трёх декодеров действия. Смена декодера не дала
устойчивого управления. Отчёты и снимки скриптов находятся в ignored `runs/`.
Старые демонстрационные веса не заменялись; GUI/Unreal не проверялись.

Побитово проверены общие начальные тензоры обычных вариантов, пары unit/mixed,
восстановление весов/Adam в продолжении. Между двумя продолжениями до обучения
различается только `log_std`; Adam совпадает полностью. Физический SHA не изменён.
Unit-checkpoint загружается через `myo.runtime.load_controller`, выдаёт 6 действий.
Неверный и отсутствующий SHA исходника exploration отвергаются. Старый runner
`myo.continue_compact` теперь отвергает reward_scale≠1 вместо молчаливой потери
масштаба награды; для таких опытов используется runner с восстановлением настроек.

Фактически выполнены:

```bash
.venv/bin/python3 -m pytest -q tests/test_compact_protocol.py
.venv/bin/python3 -m pytest -q
git diff --check
```

Результат целевого набора — **7 passed in 6.39 s**; полного — **79 passed,
2 warnings in 57.73 s**, с разрешённым локальным UDP loopback. Предупреждения
SB3 о Box прежние. Добавлены регрессии сохранения амплитуды/выборки/likelihood
при масштабировании latent и запрета терять reward_scale при resume.
Диагностические серии завершены по бюджету сравнения, не по критерию сходимости.

## Контролируемая диагностика PPO — 2026-11-02

Выполнены `scripts/diagnose_compact_ppo.py` (сбор 14720 шагов без обновлений,
обучение critic на фиксированных целях, измерение градиентов), две серии
`--phase online --case baseline/scaled` по 14720 шагов и исследование
`--phase exploration`, `exploration-wide`, `exploration-fast-wide`.
Это один seed 0 и прежняя физика, без teacher-меток.
Команды, методика, версии и результаты — в [ppo-diagnosis.md](ppo-diagnosis.md).

На одинаковых данных после 200 обновлений EV critic на отложенных целых
эпизодах равна 0.008 у исходного варианта и 0.836 при масштабе целей 0.01.
Контрольный PPO точно воспроизвёл все веса прежнего smoke checkpoint.
Масштабирование награды/value уменьшило насыщение critic со 100% до 10.86%,
медианная EV последних 20 rollout выросла с ≈0 до 0.222. При этом подъём
20→75° всё ещё не освоен. Общий gradient clipping на первом пакете не
оказался единственной доказанной причиной; исследование углов начальным
Gaussian-actor оказалось очень слабым. С σ≈0.135 проход доходит лишь до 18–20°.
Увеличение одной длительности или одной амплитуды шума почти не расширило
проход; при σ≈0.541 и сохранении случайной компоненты на 200 ms тот же
необученный actor достигает 146.01°. Это охват состояний, не успешное tracking.
Старые checkpoints не менялись.

Артефакты — `runs/d1_ppo_diagnosis/`, диагностические веса без v2 metadata
не предлагаются для демо. Полный набор `.venv/bin/python3 -m pytest -q`:
**77 passed, 2 warnings in 49.90 s**, с разрешённым локальным UDP loopback.
Предупреждения SB3 о диапазоне Box прежние. Тесты не устанавливают качество
управления; для него выполнены отдельные физические прогоны, описанные выше.

## Продолжение компактного PPO до результата или плато — 2026-11-02

Добавлен `myo.continue_compact`: продолжение полного PPO checkpoint с actor,
critic, состоянием Adam и счётчиком шагов, проверками после каждых двух циклов,
сохранением каждого блока и лучшего оценённого варианта. Награда, физика,
гиперпараметры и фиксированный набор упражнений этой серии не менялись.
Seed остаётся 0; обучение локальное, headless, CPU, один поток PyTorch.
Критерии готовности/плато приведены в [evaluation.md](evaluation.md).

Фактически запущены:

```bash
.venv/bin/python3 -m myo.continue_compact \
  --controller runs/d1_compact_ppo_smoke.zip \
  --output-dir runs/d1_compact_continued --block-cycles 2 --patience 3
.venv/bin/python3 -m myo.continue_compact \
  --controller runs/d1_compact_continued/block_001.zip \
  --output-dir runs/d1_compact_resumed --block-cycles 2 --patience 3
.venv/bin/python3 -m pytest -q
```

Второй запуск — восстановление после прерывания рабочей сессии. В первом
сохранён и оценён checkpoint на 44160 шагах; лог оборвался на 69376 шагах
внутри следующего блока, который не успел сохраниться. Эти несохранённые
25216 шагов не входят в восстановленную модель. Старые веса/логи/отчёты
оставлены в `runs/d1_compact_continued`, его summary отмечен `interrupted`.
Второй запуск восстановил веса и оптимизатор с 44160, начал новый физический
эпизод с seed 0; это не побитовое продолжение потерянного RNG/состояния среды.
Между последующими блоками состояние обучения сохраняется.

Отдельно проверена загрузка исходного optimizer: 13 состояний параметров,
Adam step=638, сумма норм `exp_avg` ≈0.54648, то есть накопленные моменты
не обнулены. При первом пробном старте проверка metadata остановила процесс
до обучения из-за сравнения JSON-списков с Python-кортежами; сравнение приведено
к единому JSON-представлению без ослабления проверки содержимого протокола.

Серия завершилась автоматически со статусом **`plateau`** на **161920 шагах**
(около 11 циклов набора в сохранённой цепочке весов). Три блока после 73600
не дали накопленного улучшения хотя бы 2% от опорной оценки 43.467.
Процесс обучения завершился с кодом 0; дальнейшие обновления не запускались.

Итоговые результаты:

| Шагов в checkpoint | Оценка, меньше лучше | Одиночные движения | Проходы |
| ---: | ---: | ---: | ---: |
| 14720, исходный | 81.564 | 0/12 | 0/6 |
| 44160 | 78.945 | 0/12 | 0/6 |
| 73600 | 43.467 | 0/12 | 0/6 |
| 103040 | 43.677 | 0/12 | 0/6 |
| 132480, лучший | 43.429 | 0/12 | 0/6 |
| 161920, последний | 43.482 | 0/12 | 0/6 |
| Эвристика, та же физика и упражнения | 5.791 | 12/12 | 6/6 |

Снижение средней ошибки на 73600 не означает освоение команды: при подъёме
и опускании рука заканчивает около 14.85°, возле нижнего упора. На 44160 подъём
с trigger=1/capacity=1 заканчивался на 21.33°, опускание — на 143.71°.
У лучшего варианта оба движения заканчиваются на 14.894°, у последнего —
на 14.835°. Все оценённые эпизоды завершены, но это не успешное управление:
в последней оценке пик реакции нижнего joint limit достиг 91.93 Nm.

Лучший checkpoint — `runs/d1_compact_resumed/best.zip`, копия `block_003.zip`,
в паре с `.zip.metadata.json`; SHA:
`dee59144ff0cbfb3ab849ae93eec666b81efc518f1323bc53e4bfcb70841b57b`.
Последний `block_004.zip` имеет SHA:
`054a61ecbcf3ee25f75334ff5b85503d2278882861cd1a9c95bd67b9b34df71b`.
Оба проверены через `myo.runtime.load_controller`: hash/контракт совместимы,
обычный actor выдаёт шесть возбуждений. Это проверка загрузки, не качества
поведения. Все веса и отчёты остаются локальными в ignored `runs/`.
`runs/d1_compact_resumed/summary.json` содержит критерии, историю и причину
остановки; `evaluation_000.json`…`evaluation_004.json` — полные сценарии.

Для диагностики без обновлений весов выполнены локальные скрипты:

```bash
.venv/bin/python3 /tmp/myo_probe_compact_policy.py \
  runs/d1_compact_ppo_smoke.zip runs/d1_compact_continued/policy_probe_parent.json
.venv/bin/python3 /tmp/myo_probe_compact_policy.py \
  runs/d1_compact_continued/block_001.zip runs/d1_compact_continued/policy_probe_001.json
.venv/bin/python3 /tmp/myo_probe_compact_policy.py \
  runs/d1_compact_resumed/block_001.zip runs/d1_compact_resumed/policy_probe_001.json
.venv/bin/python3 /tmp/myo_probe_compact_policy.py \
  runs/d1_compact_resumed/best.zip runs/d1_compact_resumed/policy_probe_best.json
.venv/bin/python3 /tmp/myo_compare_compact_rewards.py
```

Копии этих экспериментальных скриптов сохранены в ignored-каталоге
`runs/d1_compact_resumed/`. Первый собирает все 2450 наблюдений первых трёх
упражнений (trigger=0.3, capacity=1), измеряет скрытые слои/выходы сохранённой
сети и отдельно сравнивает команды 20°/75°/125° при одинаковой начальной позе.
Второй сравнивает фактические слагаемые reward actor и эвристики на этих же
трёх упражнениях; он не использует действия эвристики как обучающие метки.

Измеренные признаки проблемы:

- Уже у исходного checkpoint весь второй Tanh-слой critic имеет `abs(output)>0.99`.
  На 2450 состояниях предсказанная ценность меняется лишь от −33.19603 до
  −33.19402. На 44160 — от −81.01779 до −81.01753, на 73600 — от −99.20367
  до −99.20338. В журнале PPO `explained_variance` близка к нулю.
  У выбранного лучшего checkpoint на 132480 этот признак сохраняется:
  99.85% значений первого скрытого слоя critic и 100% второго имеют
  `abs(output)>0.99`, ценность лежит в диапазоне −147.69704…−147.69688.
- Во всех 916 записях `explained_variance` возобновлённой серии абсолютное
  значение <0.001. Из 920 PPO rollout в 877 обновления закончились раньше
  десяти эпох из-за ограничения KL. Это остановка отдельного обновления,
  не причина завершения всего процесса: вся серия остановилась по оценке поведения.
- На 44160 детерминированные возбуждения BICshort и всех трёх трицепсов
  обрезаются до нуля на всём измеренном участке. На 73600 режим изменился:
  BIClong/BRA обнулены, TRImed обрезается до 1 примерно на 97.5% наблюдений.
  Это режимы почти постоянного управления, а не успешное слежение за целью.
  У лучшего checkpoint BIClong и TRImed равны 1 на всём измеренном участке,
  BRA равна 0, TRIlat обрезается до 1 примерно на 91.9% наблюдений.
- На 73600 средняя награда actor за шаг равна −2.614/−2.410/−0.715 для
  прохода/подъёма/опускания, у эвристики +0.093/+0.648/+0.655 соответственно.
  Следовательно, на этих упражнениях текущая награда предпочитает правильное
  движение упору. Это не доказывает оптимальность самой функции награды.

Насыщение critic и обрезание выходов измерены; их роль в срыве оптимизации,
масштаб returns и соотношение обновлений actor/critic требуют отдельного
контролируемого опыта. Из этих данных нельзя заключать, что проблема в силе
мышц, нехватке числа нейронов или принципиальной неспособности PPO учиться.
Следующий предмет разбора — масштаб целей critic и его оптимизация, а также
параметризация распределения действий до обрезания в допустимые возбуждения.
Новые настройки в этой серии не проверялись. **Нового пригодного PPO-контроллера
для демонстрации не получено**; продолжать те же обновления после плато не стали.

Полный набор после изменения продолжения/сериализации:
**77 passed, 2 warnings in 34.30 s**, с разрешённым локальным UDP loopback.
Два предупреждения SB3 о несимметричном диапазоне Box сохранены.
Новые регрессии проверяют накопление небольших улучшений при обнаружении
плато и отказ считать готовым демо с одним проваленным направлением движения.
Прохождение тестов не заменяет оценку обученного управления.

## Компактные упражнения и награда за результат — 2026-11-02

По решению владельца добавлены `myo.train --protocol compact` и
`myo.evaluate --suite compact`: один seed 0, груз 1 kg, trigger 0.3/0.6/1.0,
начальная capacity всех мышц 1.0/0.8. Для каждого сочетания — непрерывный
проход 15°,25°,…,145°,150°,145°,…,15° (29 s) и два отдельных движения
20→75° / 75→20° (по 10 s). Всего 18 эпизодов / 294 s / 14700 шагов на цикл.
В этом запуске нет пяти минут удержания, перебора seeds или учительских меток.
Спецификация награды, допусков и команды запуска — в [evaluation.md](evaluation.md).

Физика и её hash не менялись:
`bc7cacd13f9bbe8dc777919fdd018a3e96a12ced614b3850f996c18af953e8e3`.
Сценарий управляет командами и reward; начальная масса/capacity задаются только
при reset. Сглаженная команда и её история в начале согласуются с позой.
Первая пробная серия выявила сохранение цели 75° при старте руки с 20°;
три пробных процесса остановлены до завершения, их логи сохранены как
`runs/d1_compact_*.initialization_trial.log`. После исправления вся серия
запущена заново. Эти прерванные попытки не входят в результаты ниже.

Фактически выполнены:

```bash
.venv/bin/python3 -m pytest -q tests/test_compact_protocol.py tests/test_evaluate.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo.train --config configs/d1_train.json \
  --protocol compact --steps 14700 --seed 0 --threads 1 \
  --log-std-init -2 --target-kl 0.02 --output runs/d1_compact_ppo_smoke.zip
.venv/bin/python3 -m myo.evaluate --controller heuristic \
  --config configs/d1_train.json --suite compact --seed 0 \
  --output runs/d1_compact_heuristic.json
.venv/bin/python3 -m myo.evaluate --controller runs/d1_actor_contact_weighted.zip \
  --suite compact --seed 0 --output runs/d1_compact_previous_actor.json
.venv/bin/python3 -m myo.evaluate --controller runs/d1_compact_ppo_smoke.zip \
  --suite compact --seed 0 --output runs/d1_compact_ppo_evaluation.json
```

Целевые проверки — **5 passed in 1.44 s**; полный набор —
**75 passed, 2 warnings in 38.11 s** с разрешённым локальным UDP loopback.
Предупреждения SB3 о диапазоне Box прежние. Новые регрессии проверяют
начальную команду/историю, физическую непрерывность через смену цели и учебный
лимит, отсутствие teacher-вызовов, предпочтение раннего точного достижения,
различие пересечения цели и установления, отказ считать прерванное окно успехом.

Короткий PPO-прогон завершён: **14720 шагов** (округление SB3 до 128),
**18 завершённых эпизодов** и начало следующего цикла. MLP 128×128,
без imitation и обученного normalizer; параметры и порядок наблюдений сохранены
в sidecar. SHA checkpoint:
`c7d282b88de4314439ccfa77572c6dcff1c79a6eb29223bbd9ed2ed00769fad0`.
SHA исходника протокола:
`2ea2b971dabb3e97056972f3d70566e11a27dbdea4deac6106ca859f52312776`.
Это проверка работающего обучения по reward, не подтверждение сходимости.

Эвристика и прежний imitation actor завершили все 18 сценариев без физических
или численных срывов. Условие установления: ±3°, скорость ≤5°/s, удержание до
конца текущей команды минимум 0.3 s. Отсчёт — от выдачи команды, не от достижения
сглаженной целью конечного угла. Это оценка на фиксированном учебном наборе.

| Проверка | Эвристика | Прежний `contact_weighted` |
| --- | --- | --- |
| Подъём 20→75°, trigger 1, capacity 1 | 75.00°, установление 1.44 s | 21.52°, установления нет |
| Опускание 75→20°, trigger 1, capacity 1 | 20.00°, установление 1.38 s | 17.17°, внутри допуска ±3°, но ошибка −2.83° |
| Проход вверх-вниз, trigger 1, capacity 1 | Установление на 19/29 командах | На 1/29 командах |
| Команда 150° в этом проходе | 141.75° к концу окна, контакт | 20.85°, до контакта не доходит |

Для эвристики подъёмы во всех шести сочетаниях установились за 1.44–1.46 s,
опускания — за 1.38 s. В коротком проходе некоторым доступным точкам не хватает
односекундного окна для полного установления; это измерение быстроты, не вывод
о недостижимости. Контактные 145°/150° сохраняются в отчёте вместе с нагрузками.
Прежняя сеть провалила подъём из 20° во всех шести сочетаниях; её хорошие
результаты из стартовой позы 75° не переносились на это упражнение.
Полный круг оценки занял 144.8 s у heuristic и 148.9 s у прежнего actor
при одновременном локальном выполнении с обучением; это не изолированный benchmark.

Оценка нового PPO завершена: все 18 сценариев прошли без численных/физических
срывов, но управление **не освоено**. При свежей capacity оба одиночных движения
заканчиваются около 143.3–143.4° в контакте вместо заданной цели. При capacity
0.8 подъём остаётся около 20.5°, опускание заканчивается около 143.1°.
Ни одно из 12 одиночных движений не установилось в целевом положении.
Полный круг занял 100.2 s; это один короткий опыт с seed 0, не сравнение seeds
и не доказательство невозможности обучения PPO. Checkpoint оставлен как smoke
в игнорируемом `runs/`; в демо по умолчанию он не включён. Награда прошла
поведенческие проверки раннего/точного достижения, однако её достаточность
для сходимости PPO данным коротким опытом не подтверждена.

`git diff --check` прошёл. Старый `d1_actor_contact_weighted.zip` и его
sidecar не менялись; текущий physical fingerprint по-прежнему совместим.
Следующие прогоны могут повторять тот же компактный набор с тем же seed;
никаких фоновых обучений или пяти­минутных проверок после этой серии не оставлено.

## Обучение контроллера под профильный контакт — 2026-11-01–02

По новой команде владельца начато обучение с нуля под текущую физику.
Старый `d1_actor_calibrated_transitions.zip` и его sidecar сохранены без изменений.
Первый этап — imitation того же нейросетевого actor, без обновлений PPO:

```bash
.venv/bin/python3 -m myo.train --config configs/d1_train.json \
  --steps 0 --seed 101 --threads 1 --warm-start-steps 60000 \
  --command-imitation --imitation-rounds 3 --log-std-init -4 \
  --output runs/d1_actor_contact_initial.zip > runs/d1_actor_contact_initial.log 2>&1
.venv/bin/python3 -m myo.evaluate --controller heuristic \
  --config configs/d1_train.json --suite contact \
  --output runs/d1_contact_heuristic_contact.json
.venv/bin/python3 -m pytest -q
```

Сбор начинается в позах не выше 135°, чтобы не создавать глубокое пересечение
оболочек при reset. Это только распределение начальных состояний, не новый
предел движения или команды. Высокие цели достигаются обычными мышечными
возбуждениями. Механика, teacher, задержки и состав наблюдений не изменены;
`dynamics_sha256` сохраняется равным `bc7cacd13f9bbe8dc777919fdd018a3e96a12ced614b3850f996c18af953e8e3`.
В конфигурации обучения явно записаны стандартные параметры контакта.

Добавлен `--suite contact`: достижимая цель 138°, команда 175° с возвратом
к 125°/45°, release и forced zero после высокой команды из идентичного состояния.
Эвристика завершила все четыре сценария: 138° без контакта; возврат к 44.996°;
после отпускания — 17.167°. Максимальная контактная сила в маршруте возврата
69.23 N, проникновение 4.062 mm. Сравниваемый снимок до release/forced zero
совпал. Эти результаты относятся к teacher, не к обучаемой сети.

После изменения сбора начальных состояний и сценариев оценки полный набор:
**72 passed, 2 warnings in 27.45 s**, с разрешённым локальным UDP loopback.
Первый этап завершён: 60 тыс. переходов, 66 обучающих и 7 отдельных
validation-эпизодов (6300 переходов). RMSE возбуждений по раундам:
0.00999 / 0.01974 / 0.02053; выбраны эпохи 15 / 19 / 1. Физических срывов
и выходов за учебный диапазон в сборе не было. Checkpoint:
`runs/d1_actor_contact_initial.zip`, SHA
`2654057df9764f6093a756a8b1ec37036625a3b578f025a34aeb0549c913ad63`.

Самостоятельные проверки первого checkpoint выявили неприемлемые ошибки:

| Проверка | Результат первого actor |
| --- | --- |
| 30 s, цели 75° / 120° / 125° | Конечные углы 76.10° / 117.58° / 122.76° |
| Контакт → release | К 24 s остаётся на 121.44°, возбуждение сгибателей ≈8.6% |
| То же состояние → forced zero | 17.167°; одинаковый снимок перед ветвлением |
| 300 s, цель 75° | 91.676°, capacity сгибателей ≈78.4% |
| 300 s, цель 125° | 119.781°, capacity ≈79.7% |
| Ещё 120 s после release с цели 125° | 98.706°, возбуждения ≈12.3%, capacity продолжает снижаться |

Файлы: `runs/d1_actor_contact_initial_{commands,holds,contact,endurance}.json`.
Это промежуточный неудачный actor; отсутствие численных срывов не делает его
пригодным контроллером. После этого запущено исправление:

```bash
.venv/bin/python3 -m myo.train --config configs/d1_train.json \
  --steps 0 --seed 111 --threads 1 --warm-start-steps 32000 \
  --command-imitation --imitation-rounds 2 \
  --resume-imitation runs/d1_actor_contact_initial.zip \
  --imitation-curriculum endurance_contact \
  --output runs/d1_actor_contact_refined.zip > runs/d1_actor_contact_refined.log 2>&1
```

Новый набор `endurance_contact` содержит 20 семейств: прежние удержания,
асимметричную capacity, отпускание и покой, плюс release после команд
155°/175°, возврат 175° → 125° → 45° и удержание 138° со сниженной capacity.
Высокие позы достигаются движением; состояние во время эпизода не исправляется.
Отдельные 16 тыс. validation-переходов покрывают все семейства. Отбор весов
сравнивает parent и кандидатов по удержанию, отпусканию и возврату;
точность достижения возле контакта оценивается при доступных 138°.
Награды за достижение недоступного угла через продавливание оболочек в отборе нет.
Контактные сценарии отбора частично совпадают по командам с итоговой suite;
они не являются независимой проверкой обобщения. Для дополнительной диагностики
подготовлены отдельные последовательности с массой 0.8/1.25 kg и другими целями.
После добавления набора полный тест: **72 passed, 2 warnings in 27.24 s**.
Второй этап завершён: 32 тыс. переходов, 46 train / 23 validation-эпизода.
Выбран первый раунд, эпоха 80; второй раунд ухудшил поведение и не заменил веса.
SHA `d1_actor_contact_refined.zip`:
`c6e323edd788332508231b891d0fb35211f31495660a9d1bd1f852a6c247ca1e`.
Сбор не имел физических срывов или выходов за учебный диапазон.
При этом отбор на массе 0.95 kg пропустил ошибку возврата при номинальных 1 kg.

| Проверка | Refined |
| --- | --- |
| 30 s, цели 75° / 120° / 125° | 75.909° / 118.274° / 124.470° |
| Команда 175° → 125° → 45°, 1 kg | Конечные 35.280°, tail RMSE 11.399° |
| Контакт → release | 17.167°, установившиеся возбуждения всех мышц — 0 |
| Удержание 300 s, цели 75° / 125° | 73.970° / 126.236° |
| 120 s отдыха после обоих удержаний | 17.167°, возбуждения — 0 |
| Слабый триггер 0.1, цель 105° | 17.167°; поведение не освоено |

Затем увеличен вес слабого триггера в supervised loss:

```bash
.venv/bin/python3 -m myo.train --config configs/d1_train.json \
  --steps 0 --seed 121 --threads 1 --warm-start-steps 16000 \
  --command-imitation --imitation-rounds 1 \
  --resume-imitation runs/d1_actor_contact_refined.zip \
  --imitation-curriculum endurance_contact --imitation-weak-weight 6 \
  --output runs/d1_actor_contact_weighted.zip > runs/d1_actor_contact_weighted.log 2>&1
```

16 тыс. переходов, 23 train / 23 validation-эпизода. Выбрана эпоха 35.
SHA `280f68770913f73002e8f8913dd02d25de09e5f396dbb90677ef66045909836c`.
Сила, утомление, teacher и runtime не менялись. Полный тест после добавления
весов: **72 passed, 2 warnings in 28.92 s**.

| Проверка | Weighted |
| --- | --- |
| 30 s, цели 75° / 120° / 125° | 75.593° / 104.870° / 121.438° |
| Команда 175° → 125° → 45°, 1 kg | 142.82° / 121.30° / 46.31° в моменты 10/20/30 s |
| Обычное опускание к 45° | 45.184° |
| Слабый триггер 0.1, цель 105° | 89.026°; улучшение, но цель не достигнута |
| Удержание 300 s, цели 75° / 125° | 74.541° / 123.495° |
| Release после контакта и после обоих удержаний | 17.167°, установившиеся возбуждения — 0 |

Оба checkpoint проверены `myo.evaluate` с suites `holds`, `commands`,
`extension`, `contact`, `endurance`; JSON и логи лежат под соответствующим
префиксом `runs/d1_actor_contact_{refined,weighted}_*`.
Диагностика `runs/check_contact_novel_commands.py` с массами 0.8/1.25 kg выявила
крупные ошибки промежуточных целей при обоих actor, хотя отпускание сработало.
Это не контроллер произвольной нагрузки или начальной позы.

Для `weighted` также выполнен offscreen-прогон:

```bash
MUJOCO_GL=egl .venv/bin/python3 runs/render_contact_actor.py \
  --controller runs/d1_actor_contact_weighted.zip --output runs/contact_weighted_viewer_qa
```

Просмотрены четыре кадра: удержания 125°/138°, контакт при команде 155° и
расслабленная рука. Все позы получены действием сети. Полное интеграционное
состояние MuJoCo и телеметрия до/после каждого рендера совпали.
Отчёт — `runs/contact_weighted_viewer_qa/report.json`. У нижней позы кисть
слегка перекрывает первую строку экранной подсказки; интерактивный GLFW-сеанс
и темп реального времени этой проверкой не подтверждены.

Исторически второй этап использовал равные веса примеров; поддержка
`--imitation-weak-weight` для контактного набора появилась на третьем этапе.
Кроме того, ниже исправлено распределение начальных поз и отбор. Поэтому
повтор прежних команд на новом коде не обязан воспроизводить прежние веса.
Индивидуальные настройки и результаты каждого запуска записаны в sidecar.

После регрессии цели 120° в контактный набор добавлены 30-секундные подъёмы
75° → 120°/125° в каждом цикле вместо старта сразу в конечной позе.
Отбор теперь также проверяет эти подъёмы и возврат из контакта при 1 kg,
сохраняя проверку возврата при 0.95 kg. Это диагностические сценарии отбора,
не независимый тест обобщения. Четвёртый этап:

```bash
.venv/bin/python3 -m myo.train --config configs/d1_train.json \
  --steps 0 --seed 131 --threads 1 --warm-start-steps 36000 \
  --command-imitation --imitation-rounds 2 \
  --resume-imitation runs/d1_actor_contact_weighted.zip \
  --imitation-curriculum endurance_contact --imitation-weak-weight 2 \
  --output runs/d1_actor_contact_balanced.zip > runs/d1_actor_contact_balanced.log 2>&1
```

После изменения свежих подъёмов и отбора выполнен полный
`.venv/bin/python3 -m pytest -q`: **72 passed, 2 warnings in 24.42 s**,
с разрешённым локальным UDP loopback. Предупреждения SB3 о диапазоне Box
прежние.

Четвёртый процесс завершился с кодом **143** до сохранения checkpoint;
причина по логу не установлена. Первый раунд собрал 18 тыс. переходов и
проверил кандидатов. Подъёмы 120°/125° улучшились примерно до 119.6°/124.9°,
слабый триггер — до 101.5°, но возврат к 45° заканчивался около 50.2°/50.7°.
Из-за двух превышений порога 5° критерий отбора сохранил parent, несмотря
на его гораздо большую ошибку подъёма к 120°. Это ограничение нынешнего
порогового отбора; пригодность нового распределения данных не доказана.
Второй раунд не завершён, `runs/d1_actor_contact_balanced.zip` не создан.
Исправленный сбор и отбор остаются в коде, но успешного обучения с ними
не заявляем. Запущенных фоновых обучений к завершению работы нет.

Для исследования оставлен **`runs/d1_actor_contact_weighted.zip`** со своим
sidecar, SHA указан выше. Это третий завершённый этап, не результат четвёртого.
Все три завершённых этапа собрали 108 тыс. обучающих переходов; не каждый
раунд вошёл в выбранные веса. Веса не включены в git и не заменяют старую
отслеживаемую пару. Эвристика остаётся запуском по умолчанию.

Фактически выполненные проверки `weighted` (для `refined` использован тот же
набор suites с соответствующими именами файлов):

```bash
.venv/bin/python3 -m myo.evaluate --controller runs/d1_actor_contact_weighted.zip \
  --suite holds --output runs/d1_actor_contact_weighted_holds.json
.venv/bin/python3 -m myo.evaluate --controller runs/d1_actor_contact_weighted.zip \
  --suite commands --output runs/d1_actor_contact_weighted_commands.json
.venv/bin/python3 -m myo.evaluate --controller runs/d1_actor_contact_weighted.zip \
  --suite contact --output runs/d1_actor_contact_weighted_contact.json
.venv/bin/python3 -m myo.evaluate --controller runs/d1_actor_contact_weighted.zip \
  --suite extension --output runs/d1_actor_contact_weighted_extension.json
.venv/bin/python3 -m myo.evaluate --controller runs/d1_actor_contact_weighted.zip \
  --suite endurance --output runs/d1_actor_contact_weighted_endurance.json
.venv/bin/python3 runs/check_contact_novel_commands.py \
  --controller runs/d1_actor_contact_weighted.zip \
  --output runs/d1_actor_contact_weighted_novel.json
```

Первые четыре suite вызывались последовательно через `myo.evaluate.main` в
одном Python-процессе с ровно этими аргументами; endurance и отдельная диагностика
— указанными CLI. Все 21 сценарий `weighted` завершились без физических и
численных срывов. Это не значит, что все команды выполнены: ошибки приведены
выше. Снимки ветвления совпали в `commands`, `contact`, `extension`.
За последние 5 s трёх обычных удержаний std угла — 0.0252°/0.0226°/0.0217°;
малый разброс вокруг неверной цели 120° не означает точного управления.
В пятиминутных опытах capacity сгибателей снизилась до 77.24%/79.95%,
после ещё 120 s отдыха — 95.73%/96.27%; возбуждения всех мышц равны нулю.
Это результаты нашей capacity proxy, не подтверждение человеческой физиологии.
Пик силы контакта в возврате — 66.64 N, проникновение — 3.93 mm.
В `extension_release` есть пик реакции нижнего ограничения 8.89 Nm;
даже успешное расслабление не является доказательством безударного торможения.

Проверены четыре рендера, загрузка checkpoint с текущим fingerprint и
`git diff --check`. Старый отслеживаемый actor сохранил SHA
`816cad5d63ba518d0f4e4771bc199e13897de2ecc10100739845080c9ef18399`.
Произвольные грузы, другие наклоны стенда и полная D1-приёмка этой серией
обучения не закрыты.

## Профили контактных оболочек и настройка мягкости — 2026-11-01

Два прежних эллипсоида заменены непрерывными выпуклыми профилями: 7 сечений
плеча и 6 предплечья, по 16 вершин в каждом. Продольные размеры опираются
на положения локтя и запястья MyoArm; поперечные размеры — инженерные оценки.
Есть утолщение плеча, сужение у локтя и к запястью. Область общего шарнира
исключена из контактного объёма. Это собственная процедурная форма, без
импорта BodyParts3D. Параметры записаны в локальный manifest.
Одна пара мешей может дать несколько точек контакта; телеметрия суммирует
нормальные силы и отдельно считает момент относительно локтя.

Мягкость задают `tissue_contact_time_constant=0.03 s` и
`tissue_contact_transition_depth=0.015 m`. В `d1_myoarm_soft.json` первое
значение увеличено до 0.05 s, остальное совпадает. Импеданс контакта
возрастает от 0.1 до 0.95 по мере проникновения. Это податливость решателя,
зависящая от геометрии и эффективной инерции, а не измеренный материал тканей.
Меши допускают пересечение, но визуально не сплющиваются. Активация их форму
или мягкость пока не меняет; верхний joint limit остаётся `+inf`.

Контроллер не адаптировали. Новый `dynamics_sha256`:
`bc7cacd13f9bbe8dc777919fdd018a3e96a12ced614b3850f996c18af953e8e3`.
Сохранённый actor и sidecar не менялись, SHA весов остаётся
`816cad5d63ba518d0f4e4771bc199e13897de2ecc10100739845080c9ef18399`;
загрузка отклонена проверкой несовместимой физики. Во всех опытах ниже
использована прежняя эвристика либо явно заданное диагностическое воздействие.

Фактически выполнены:

```bash
.venv/bin/python3 scripts/prepare_myoarm.py
.venv/bin/python3 -m pytest -q tests/test_myoarm_geometry.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 scripts/audit_tissue_contact.py --output runs/profile_contact_response.json
.venv/bin/python3 scripts/audit_tissue_contact.py --config configs/d1_myoarm_soft.json --output runs/profile_contact_response_soft.json
.venv/bin/python3 runs/probe_profile_contact_geometry.py
.venv/bin/python3 runs/check_profile_contacts.py
MUJOCO_GL=egl .venv/bin/python3 runs/render_profile_contacts.py
git diff --check
```

Assets пересобраны из 39 проверенных локальных файлов без загрузки.
Геометрические проверки: **10 passed in 0.99 s**. Первый полный запуск
в sandbox дал 71 passed и отказ существующего UDP-теста открыть сокет;
повтор с разрешённым loopback: **72 passed, 2 warnings in 18.05 s**.
Предупреждения SB3 о диапазоне действий прежние. Проверены отсутствие добавленной
массы/инерции, свободные позы 15–135°, предотвращение прежнего проворота,
отдельная телеметрия контакта и влияние обоих параметров мягкости.

Первое касание — **139.9289°**, одинаковое в стандартном и мягком вариантах.
В `audit_tissue_contact.py` фиксируются поза и скорость, отключены гравитация,
приложенные силы и собственное демпфирование сустава; мышцы и интегрирование
не вызываются. Это мгновенный отклик решателя, не равновесие удержания:

| Максимальное проникновение | Сила при нулевой скорости, стандарт | Более мягкий вариант |
| --- | ---: | ---: |
| 0.302 mm | 0.34 N | 0.12 N |
| 2.412 mm | 6.50 N | 2.34 N |
| 4.980 mm | 37.96 N | 13.66 N |
| 7.692 mm | 132.11 N | 47.56 N |

При той же глубине 4.980 mm и скорости сгибания локтя 0.5/2 rad/s
стандартный вариант дал 95.12/266.62 N, более мягкий — 47.96/150.86 N.
Это угловая скорость, не линейная скорость каждой точки контактного пятна.
Сила возрастает с глубиной и скоростью сближения; 15 mm — масштаб перехода,
а не запрет на большее проникновение.

Механическое воздействие +100 N·m в течение 3 s от 125° дало максимум
**153.482°**, итог **146.479°**, пик проникновения 18.20 mm и суммарной силы
3370 N. Без оболочек тот же скелет в регрессии проходит за 180°.
Внешний момент используется только в тестовом стенде, не в runtime-контроллере.

Восемь непрерывных 16 s сценариев с 1 kg (`profile_contact_behavior.json`)
завершились без физического или численного сбоя: удержание и маршрут
75° → 175° → 75° → release для обычного, горизонтального, перевёрнутого
и более мягкого вариантов. Обычный/горизонтальный удержали 125°,
перевёрнутый — 75°. В контактной фазе обычный вариант установился на
142.875° с 3.557 mm и 35.09 N, более мягкий — на 143.787° с 4.788 mm
и 33.97 N. После обратной команды в 12 s углы всех вариантов лежали
в 74.976–75.523°, контакта не было. После release в 16 s обычный/мягкий
оказались около 17.167°, горизонтальный — 16.172°, перевёрнутый — 121.378°
из-за другого направления тяжести. Достижение запрошенных 175° не заявляется.

Максимальное возбуждение сгибателей `[1,1,1,0,0,0]` на 3 s от 125°
проверено с обычной моторной задержкой, при одинаковом периоде управления 20 ms:

| Контакт | dt | Максимальный угол, выборка 50 Hz | Конечный угол | Пик силы по подшагам | Пик проникновения |
| --- | ---: | ---: | ---: | ---: | ---: |
| Стандартный | 1 ms | 147.338° | 145.280367° | 774.30 N | 9.668 mm |
| Стандартный | 0.5 ms | 147.384° | 145.280373° | 775.25 N | 9.753 mm |
| Более мягкий | 1 ms | 150.375° | 146.287505° | 628.18 N | 13.756 mm |
| Более мягкий | 0.5 ms | 150.439° | 146.287511° | 627.58 N | 13.842 mm |

Реакция joint limit во всех четырёх опытах нулевая. Эти близкие результаты
подтверждают проверенные режимы, не произвольные нагрузки и шаги интегрирования.

EGL сохранил десять кадров трёх ориентаций и двух вариантов контакта,
с оболочками и без: `runs/profile_contact_viewer_qa/report.json`.
Дополнительный inline EGL-прогон сохранил позы 15/75/135° и
`pose-checks.json`. Визуально проверены форма и совмещение оболочек, обычный
и мягкий контакт, три ориентации и дополнительные позы. Во всех 13 рендерах
интеграционное состояние и телеметрия совпали до/после изображения.
В позе 15° кисть частично заходит под нижнюю подсказку управления;
реальное GLFW-окно и пользовательский ввод этим offscreen-прогоном не проверены.

## Первые эллипсоидные оболочки и ориентация плеча — историческая проверка

В предыдущую механику были добавлены две фиксированные контактные оболочки плеча
и предплечья. Существующие кости и направляющие сухожилий остаются неконтактными;
одна явно заданная пара эллипсоидов создаёт нормальную податливую реакцию MuJoCo.
Массы и инерции исходных сегментов не изменились. Предел локтя сверху остаётся
`+inf`, команды не обрезаются. Размеры и жёсткость оболочек — инженерные
приближения, пока без зависимости от активации/длины мышц и без подгонки к человеку.

По указанию владельца **контроллер не адаптировался и обучение не выполнялось**.
Все прогоны этого раздела используют прежнюю эвристику либо явно обозначенный
механический стресс-тест. Нейросетевые результаты следующих разделов относятся
к прежней физике без контактных оболочек.
Новый `dynamics_sha256`:
`b461fd6ae194e3c815721d480ac5034235b8d0160bd8582f085d30d044e2daf1`.
Сохранённые веса `d1_actor_calibrated_transitions.zip` не изменены (SHA
`816cad5d63ba518d0f4e4771bc199e13897de2ecc10100739845080c9ef18399`);
их загрузка проверена и отклонена с `Policy dynamics source/model fingerprint
does not match this installation.` Метаданные не переписаны для обхода проверки.

Фактические команды:

```bash
.venv/bin/python3 scripts/prepare_myoarm.py
.venv/bin/python3 -m pytest -q tests/test_myoarm_geometry.py
.venv/bin/python3 -m pytest -q
.venv/bin/python3 runs/probe_tissue_contact_geometry.py
.venv/bin/python3 runs/check_contact_mounts.py
MUJOCO_GL=egl .venv/bin/python3 runs/render_contact_mounts.py
git diff --check
```

Assets пересобраны из 39 локальных проверенных файлов без загрузки.
Геометрические регрессии: **9 passed in 0.78 s**; полный набор:
**71 passed, 2 warnings in 14.90 s**. Для существующей UDP-проверки полному
набору разрешён локальный loopback. Дополнительный тест покрывает прежнее
сквозное прохождение, нулевое добавление массы/инерции, свободные позы до 140°,
разделение контакта и joint limit. В существующую проверку путей добавлена
инверсия крепления: локальные длины/плечи силы сохраняются, гравитационный
момент меняет знак. Переключение C проверено без изменения физического состояния.

Геометрический отчёт `runs/tissue_contact_geometry.json`: первый контакт
**144.8021°**. Механический стресс-тест подаёт постоянные +100 N·m от 125°
на 3 s только для проверки коллизии, это не обход мышц в runtime-контроллере.
С контактом максимум **155.879°**, итог **147.236°**; без оболочек тот же
скелет проходит за 180°. Ударный пик проникновения 23.87 mm, нормальной силы
2452 N; установившееся проникновение менее 10 mm. Эти числа характеризуют
выбранную мягкость solver, не измеренную компрессию или пределы тканей человека.

`runs/contact_mount_behavior.json` содержит шесть непрерывных 16 s опытов
с 1 kg и свежей исходной capacity. Для удержаний цель меняется с 2 s:

| Крепление | Цель | Угол в 16 s | Активация трицепса в 16 s | Контакт |
| --- | ---: | ---: | ---: | --- |
| Обычное, 30° | 125° | 124.9971° | практически 0 | Нет |
| Горизонтальное, 90° | 125° | 125.0009° | 8.755% | Нет |
| Перевёрнутое, 210° | 75° | 75.0061° | 14.807% | Нет |

В контактных опытах задавались 75° → 175° с 2 s → 75° с 8 s → trigger=0
с 12 s. Максимальные углы обычного/горизонтального/перевёрнутого вариантов —
145.619°/145.973°/145.510°. При удерживаемой высокой команде в 8 s контакт
даёт около 19.96 N и 0.174 mm проникновения; конечные углы удержания около
144.887°. В 12 s после обратной команды получены 75.506°/74.972°/75.539°
без контакта. После отпускания в 16 s — 17.167°/16.172°/121.378°;
в перевёрнутой постановке тяжесть сгибает руку, поэтому расслабление
не приводит к обычному нижнему покою. Все шесть опытов завершены без
численных/физических сбоев; достижение недоступных 175° не заявляется.

Отдельно выполнены 3 s максимального возбуждения сгибателей `[1,1,1,0,0,0]`
от 125° с обычными моторной задержкой и активацией. При dt=1/0.5 ms и одном
периоде управления 20 ms максимумы угла **149.5951°/149.6721°**, итоговые
**145.825050°/145.825056°**. Пики нормальной силы **800.86/800.92 N**,
проникновения **10.351/10.601 mm**. Joint-limit reaction равна нулю;
контактный момент регистрируется отдельно. Это проверка двух конкретных
шагов/воздействий, не гарантия устойчивости при любом ударе или любой нагрузке.

Исходные прикрепления проверены отдельно: все шесть путей, sites, wraps
и sidesite совпадают с pinned upstream. Исходная 38-DOF цепь при остальных
`q=0` и производная 1-DOF дали расхождения длин не более 2.22e−16 m,
плеч силы 1.11e−16 m; численная производная `−dL/dq` согласуется до
7.41e−11 m. Аудиты: `runs/attachment_audit_angles.json` и
`runs/mount_orientation_audit.json`. Исчезновение рычага трицепса на
155–162° присутствует в самом исходнике вне его диапазона 0–130.004°.
Подробная интерпретация — [calibration.md](calibration.md).

EGL-проверка сохранила и визуально проверила восемь кадров трёх ориентаций
и контакта, с оболочками и без них: `runs/contact_mount_viewer_qa/report.json`.
Камера охватывает руку/груз, оболочки совмещены с сегментами. Для каждого кадра
интеграционное состояние и телеметрия до/после рендера совпали.
После обнаруженного перекрытия подписи грузом контактный HUD сокращён,
кадры повторно отрисованы. Это offscreen-проверка: реальные GLFW-события
и интерактивное окно не проверялись. Кожа, Flesh и деформируемые контактные
оболочки пока не реализованы.

## Снятие верхнего предела и предварительная калибровка — 2026-11-01

Следующий раздел описывает физику до добавления контакта тканей. Все результаты ниже раздела
«Исправление нейросетевого демо» исторические: checkpoints `d1_actor_initial.zip`
и `d1_actor_corrected.zip` с ней несовместимы и отвергаются по fingerprint.
Первый новый checkpoint — `runs/d1_actor_calibrated.zip`; его обучение
завершено, но проверка обнаружила срыв при 155° (подробности ниже).
Итоговый локальный checkpoint — `runs/d1_actor_calibrated_transitions.zip`:
проверены удержания 75°/125° по 300 s, отпускание после них и отдельный проход
к 155° с возвращением. Резкий переход 125° → 155° остаётся неустойчивым.
Ниже сохранены результаты промежуточных этапов и итоговые измерения.

У MyoArm сохранён нижний предел 15°, верхний диапазон MuJoCo теперь действительно
`inf`. Верхняя мягкая пружина, damping и соответствующий защитный guard не
действуют. Команды 175° и 200° не обрезаются; конечная область тренировочных
целей до 158° не является ограничителем физики. Телеметрия сериализует
отсутствующую верхнюю границу как `null`, а разворот мышечных путей отмечает
предупреждением без изменения движения.

```bash
.venv/bin/python3 scripts/prepare_myoarm.py
.venv/bin/python3 -m myo.calibration --fit-source-width \
  --fit-reference configs/adult_male_strength_reference.json \
  --output runs/arm_calibration_fit.json
.venv/bin/python3 -m myo.calibration --config configs/d1_myoarm.json \
  --output runs/arm_calibration.json
.venv/bin/python3 -m myo.fatigue_calibration --output runs/fatigue_calibration.json
.venv/bin/python3 -m myo.evaluate --controller heuristic --suite extension \
  --output runs/calibrated_heuristic_extension.json
.venv/bin/python3 -m myo.evaluate --controller heuristic --suite vertical \
  --output runs/calibrated_heuristic_vertical.json
```

Assets пересобраны из 39 уже проверенных локальных файлов. Статическая настройка
дала ширину force–length **0.7949002963** и масштаб активной силы сгибателей
**1.3692387128**. Отложенные ошибки относительно исходной кривой MyoArm:
0.440/0.903 N·m для сгибания/разгибания. Мужской ориентир 64.3 N·m при 90°
использован только для предварительного масштаба: это средний момент 10 s MVC
из несовпадающего по позе протокола, а не подтверждённая антропометрия.
Масштаб не меняет пассивную упругость и жёсткость сухожилия.

Коэффициенты утомления F=0.00912 s⁻¹, R=0.00094 s⁻¹ и множитель восстановления
в покое 15 перенесены из литературы. Отдельный идеальный тест даёт времена
565.4/115.7/27.77 s для 20/50/80% MVC. На отложенной сетке опубликованной
кривой MAPE=28.69%; ошибки особенно велики около максимальной нагрузки.
Это редуцированный proxy с прежней активацией, не полная модель 3CC-r и
не индивидуальная человеческая валидация. Источники и допущения:
[calibration.md](calibration.md).

Старая сеть при release сохраняла около 8% возбуждения сгибателей в покое.
При этом точные 15° с полным триггером требуют активного момента против
нижней пассивной опоры. Поэтому отпускание и точная команда оцениваются отдельно.
У teacher убрана постоянная коактивация командного режима; торможение у нижней
границы масштабируется по доступному мышечному моменту. Демпфирование tracking
увеличено с 9 до 14, что устранило перелёт при отдельном подъёме от 75° к 155°.
Это не устраняет описанный ниже срыв при смене цели 125° → 155°.
Все команды проходят прежние 40 ms задержки и динамику активации.

Heuristic, 1 kg, начальная capacity=1, непрерывные 30 s:

| Сценарий | Конечный угол | Возбуждение последних 5 s |
| --- | ---: | --- |
| Точная команда 15°, g=1 | 15.001° | Трицепсы 6.21%, сгибатели 0 |
| Отпускание g=0 | 17.167° | Все мышцы 0 |
| Принудительный ноль | 17.167° | Все мышцы 0 |
| Цель 140° | 139.999° | Сгибатели 5.95%, трицепсы 0 |
| Цель 150° | 150.000° | Сгибатели 2.14%, трицепсы 0 |
| Цель 155° | 155.000° | Сгибатели 0, трицепсы 11.62% |
| Цель 175° — экстраполяция | 188.941° | Разворот путей; цель не удерживается |

Гравитационный момент меняет знак при **146.113°**, необходимый активный
момент с учётом пассивных мышц — при **153.490°**. Плечи трицепса меняют
знак при **161.546°**: за этой точкой нельзя интерпретировать движение как
анатомически правильное. При 160° уже не хватает разгибающего момента даже
для статического удержания 1 kg. Это ограничение исходной геометрии, не новый
программный предел. Дополнительный stress-прогон teacher: 155° удерживаются
при 1 kg/capacity=0.5, но при 2 kg происходит перелёт даже со свежими мышцами.
Отпускание при 2 kg/capacity=0.5 задействует нижний численный упор.

```bash
MUJOCO_GL=egl .venv/bin/python3 runs/render_extended_rom_qa.py
```

Проверены семь EGL-кадров в `runs/extended_rom_viewer_qa`: фиксированные
140°/150°/155°/175° и фактические удержания 140°/150°/155°. Пути конечны,
нативные кости и процедурные ткани отображаются; при 175° видны складывание
за область исходной анатомии и предупреждение. При 155° в движении получены
нулевые сгибатели и 11.38% активации трицепса. На больших углах кисть и груз
частично перекрывают левую панель. Кожа и реальное GLFW-окно здесь не проверялись.

```bash
.venv/bin/python3 -m myo.evaluate --controller heuristic --suite endurance \
  --output runs/calibrated_heuristic_endurance.json
.venv/bin/python3 -m pytest -q
```

Оба опыта удержания 1 kg завершили 300 s без срыва, затем продолжили ещё
120 s после отпускания без reset. Heuristic:

| Цель | Угол в 300 s | Capacity сгибателей в 300 s | После 120 s отдыха |
| --- | ---: | ---: | ---: |
| 75° | 74.9958° | 77.39% | 95.79% |
| 125° | 124.9973° | 80.37% | 96.34% |

В конце отдыха оба опыта имеют угол 17.1674° и нулевое возбуждение всех мышц.
Это проверка связанной симуляции и teacher; она не заменяет отдельного опыта
с нейросетью или независимой проверки человеческой выносливости.

Полный pytest после изменений физики и телеметрии: **68 passed, 1 warning
in 28.31 s**. Для UDP-теста был разрешён только необходимый локальный loopback;
предупреждение SB3 о диапазоне действий прежнее. Проверены отсутствие верхнего
упора и обрезания команды, конечный учебный диапазон, JSON `null`, разворот
путей, статический баланс, независимость пассивных тканей от масштаба активной
силы, сохранение задержки перед ускоренным восстановлением. Прямой запуск
загрузчика подтвердил отказ для обоих старых checkpoints. Текущий
`dynamics_sha256`: `23f8a1313177949964d42521e2fa721a0500a5cc733d92a551c77ed4ff143fa8`.

### Первый actor на новой физике: обнаруженный срыв

```bash
.venv/bin/python3 -m myo.train --config configs/d1_train.json \
  --steps 0 --seed 7 --warm-start-steps 60000 \
  --command-imitation --imitation-rounds 3 --log-std-init -4 \
  --output runs/d1_actor_calibrated.zip > runs/d1_actor_calibrated.log
.venv/bin/python3 -m myo.evaluate --controller runs/d1_actor_calibrated.zip \
  --output runs/d1_actor_calibrated_evaluation.json
.venv/bin/python3 -m myo.evaluate --controller runs/d1_actor_calibrated.zip \
  --suite holds --output runs/d1_actor_calibrated_holds.json
.venv/bin/python3 -m myo.evaluate --controller runs/d1_actor_calibrated.zip \
  --suite extension --output runs/d1_actor_calibrated_extension.json
.venv/bin/python3 -m myo.evaluate --controller runs/d1_actor_calibrated.zip \
  --suite vertical --output runs/d1_actor_calibrated_vertical.json
```

60 тыс. переходов, 85 обучающих и 9 отдельных validation-эпизодов;
6300 validation-переходов, без обновлений PPO. Normalizer заморожен после
первого раунда. RMSE возбуждения по раундам 0.01330 → 0.01190 → 0.01128.
Сбор завершил 7/5/8 тренировочных и 2 validation-эпизода при выходе за область
160°; физической остановки в runtime это не добавляет. Для нулевых teacher-label
использован raw target −0.03: точный ноль получается прежним clipping PPO,
без нового runtime-порога или умножения на триггер.

Отпускание и пассивный покой заканчиваются на 17.16744° с точным нулём
возбуждения. Обычные 30 s удержания заканчиваются на 78.52°/120.51°/125.94°
для целей 75°/120°/125°. Но точные 15° дают 20.08° с лишней активацией
сгибателей, а trigger=0.1 не поднимает руку. При 155° сеть сначала оказывается
на 153–154°, затем раскачивается и проходит через область разворота путей:
максимум около 371°. Это **не успешный нейросетевой проход вертикали**,
несмотря на малую ошибку на validation и успех teacher.

`MUJOCO_GL=egl .venv/bin/python3 runs/render_calibrated_actor.py` воспроизвёл
срыв на 10 s: 208.66°, максимум 370.50°. Кадр и след лежат в
`runs/extended_rom_viewer_qa/actor-155.*`; рендер показывает фактическое состояние
и предупреждение, а не фиксирует позу. Длительная оценка первого кандидата
прервалась с кодом 143 без итогового отчёта и не считается выполненной.

### Второй actor: короткие проверки проходят, длительное удержание — нет

```bash
.venv/bin/python3 -m myo.train --config configs/d1_train.json \
  --steps 0 --seed 17 --warm-start-steps 20000 --command-imitation \
  --imitation-rounds 2 --resume-imitation runs/d1_actor_calibrated.zip \
  --output runs/d1_actor_calibrated_refined.zip
.venv/bin/python3 -m myo.evaluate --controller runs/d1_actor_calibrated_refined.zip \
  --suite endurance --output runs/d1_actor_calibrated_refined_endurance.json
```

Дополнительно выполнены suites `commands`, `holds`, `extension`, `vertical`
и соседние цели 154°/156°. Продолжение обучения проверяет контракт родительского
checkpoint, сохраняет normalizer и записывает SHA родителя; физика не меняется.
20 тыс. новых переходов, 22 обучающих и 13 validation-эпизодов; выбор снимков
actor по отдельным замкнутым сценариям, выбранные эпохи 90/60.
RMSE возбуждения 0.006732/0.005841. Отрицательная raw-метка −0.03 применяется
только к строкам, где teacher расслабляет все мышцы. Runtime остаётся одной сетью
с прежним clipping выходов и обычной задержкой команд.

На 30 s цель 155° достигается: итог 155.456°, максимум 156.744°, RMS ошибки
последних 5 s 0.734°. Среднее возбуждение сгибателей в этом окне около 0.002,
трицепсов 0.146. Цель 156° уже срывается; это ошибка controller до разворота
геометрии при 161.55°, а не свидетельство успешного контроля всего диапазона.
Команда 15° и отпускание заканчиваются в пассивном положении 17.167° с нулевым
возбуждением. При отпускании максимум proxy реакции сустава 2.02 N·m против
41.23 N·m при принудительном нуле, но максимум силы сухожилия выше:
430.6 против 147.2 N. Это сравнение конкретного опыта, не гарантия защиты.

Непрерывная проверка 420 s выявила регрессию, отсутствовавшую в коротких тестах:

| Цель | Угол в 300 s | Capacity сгибателей | Capacity после 120 s release |
| --- | ---: | ---: | ---: |
| 75° | 77.488° | 77.81% | 95.87% |
| 125° | 104.039° | 79.29% | 72.74% |

Второй опыт после release сохраняет около 11% возбуждения сгибателей;
мышцы продолжают утомляться. Этот checkpoint не является исправлением
длительного удержания и расслабления после высокой цели.

Причина подтверждена контрфактическими пробами
`runs/probe_refined_fatigue_shortcut.py`: при одинаковой позе 125°, capacity
сгибателей 0.8, одинаковых силах и нулевой активации трицепсов изменение только
их capacity 0.8 → 1 снижает команду сгибателям 0.09632 → 0.08596, тогда как
teacher сохраняет 0.10263. Ранее все мышцы начинали учебный эпизод с одинаковой
capacity; длительное удержание создаёт не представленный в обучении профиль.
Отдельно, в покое 18.64° с trigger=0 смена сохранённой цели 75° → 125° даёт
команду 0 → 0.08437 при неизменном физическом состоянии; teacher выдаёт ноль.
Нужны примеры как раздельного утомления, так и покоя после высоких целей.

После добавления продолжения imitation: `.venv/bin/python3 -m pytest -q` —
**68 passed, 2 warnings in 22.79 s**; предупреждения SB3 о диапазоне действий.
SHA второго checkpoint:
`7d50bc93a505fadf4a43a47c7fe0a13f2a5961344d6417d8696763d529d5112f`.

### Асимметричное утомление: исправление выборки

```bash
.venv/bin/python3 -m myo.train --steps 0 --seed 27 --threads 1 \
  --warm-start-steps 20000 --command-imitation --imitation-rounds 2 \
  --resume-imitation runs/d1_actor_calibrated_refined.zip \
  --imitation-curriculum endurance --output runs/d1_actor_calibrated_balanced.zip
```

16 систематических семейств включают удержания 75°/120°/125° с capacity
сгибателей 0.75/0.85 при свежих разгибателях, отпускание после 125° и покой
с сохранёнными высокими целями. Профиль задаётся только перед первым шагом:
пересчитываются равновесие мышц и начальная сенсорная история, задержки
сохраняются. При проверке максимальная невязка равновесия 3.64×10⁻⁸ N.
В работающем эпизоде такая инициализация запрещена. Физический fingerprint
не менялся; в runtime нет подмены команды эвристикой.

36 обучающих и 18 отдельных validation-эпизодов, 10 тыс. validation-переходов.
Выбраны эпохи 80/120, RMSE возбуждения 0.003608/0.002836. В отдельных замкнутых
проверках используется промежуточный профиль 0.8/1, которого нет среди начальных
обучающих профилей. Удержание 125° и покой со старой целью исправились,
однако номинальный 30 s опыт с целью 155° вновь срывается: максимум 365.69°
(`runs/d1_actor_calibrated_balanced_nominal155.json`). Этот промежуточный
checkpoint также не подходит для итогового демо.

`.venv/bin/python3 -m myo.evaluate --controller runs/d1_actor_calibrated_balanced.zip
--suite endurance --output runs/d1_actor_calibrated_balanced_endurance.json`
завершился без физических и численных срывов. В 300 s углы 75.0765°/125.0305°,
capacity сгибателей 77.14%/80.33%. Удержание исправлено, но после 120 s release
осталось возбуждение сгибателей 1.157%/1.032%; capacity восстановилась лишь
до 87.35%/85.54%. Остаток около инженерного порога покоя 1% мешает ускоренному
восстановлению. Это ещё одна причина включать свежий покой после высоких целей.

После добавления инициализации профилей `.venv/bin/python3 -m pytest -q`:
**69 passed, 2 warnings in 23.36 s**. SHA этого checkpoint:
`6150038ddcae86c043d3089fa56d88be69cd6916be7eb9f3fd5808de153fcb1a`.

### Дополнительные состояния возле вертикали

```bash
.venv/bin/python3 -m myo.train --steps 0 --seed 37 --threads 1 \
  --warm-start-steps 30000 --command-imitation --imitation-rounds 2 \
  --resume-imitation runs/d1_actor_calibrated_balanced.zip \
  --imitation-curriculum endurance_vertical \
  --output runs/d1_actor_calibrated_balanced_vertical.zip
```

24 семейства: сохранены предыдущие удержания/отпускания, добавлены состояния
148–157° с целями 150°/154°/155°, вариацией массы/начальной скорости/capacity
и свежий покой после целей 125°/155°. Бюджет 15 тыс. переходов на раунд позволяет
охватить все 24 семейства. Loss весит состояния выше 140° в 4 раза сильнее,
слабый триггер — в 2 раза. Это веса обучения, не изменение физической силы.

64 обучающих и 32 validation-эпизода; 15 тыс. validation-переходов.
Выбраны эпохи 80/80, RMSE возбуждения 0.003828/0.004045. Во втором раунде
проверка 155° с 1.05 kg на 12 s достигает 154.99°, но номинальная проверка
1 kg на 30 s срывается до 235.45°. Остаток возбуждения при release исчезает.
Из-за срыва 155° дальнейшие suites для этого промежуточного checkpoint
не запускались. SHA:
`bc3d8aa3d9e5dd67866f0341203b7895fa9efd4b20457522523e946159937334`.

Подробная проба предыдущего `balanced` на одинаковых физических состояниях
объясняет чувствительность: в 4.5 s при 155.528° и скорости 0.082 rad/s сеть
выдаёт сгибателям 0.0140, трицепсам 0.1036, а teacher — 0 и 0.2333. После этого
ускорение ведёт в область исчезающего плеча трицепса. Небольшая ошибка
возбуждения сильных сгибателей здесь имеет большой механический эффект.
Кроме того, положительно обрезанный шум 0.012 в teacher-rollouts систематически
возбуждает даже неактивные сгибатели; это отдельная проблема выборки возле
верхней вертикальной позы.

```bash
.venv/bin/python3 -m myo.train --steps 0 --seed 47 --threads 1 \
  --warm-start-steps 30000 --command-imitation --imitation-rounds 2 \
  --resume-imitation runs/d1_actor_calibrated_balanced_vertical.zip \
  --imitation-curriculum endurance_vertical --imitation-vertical-noise 0 \
  --imitation-inactive-flexor-margin 0.01 --imitation-weak-weight 6 \
  --output runs/d1_actor_calibrated_control.zip
```

В этом опыте шум teacher при высоком угле отключён. Для неактивных сгибателей
на строках с работающими разгибателями raw target смещается до −0.01,
плавно от нуля при росте команды разгибателей до 0.05. Начало включения
трицепса не получает отрицательной метки; runtime-маски по углу/триггеру нет.
60 обучающих и 31 validation-эпизод, 15 тыс. validation-переходов;
выбраны эпохи 120/40, RMSE 0.004288/0.005870. Номинальный 30 s проход
155° завершается на 154.849°, но полное отпускание после удержания даёт
сильную коактивацию около 0.4. Инициализированный низкий покой при этом
выдаёт ноль: одного такого теста недостаточно. Этот вариант также не принят.
SHA: `b8ae84b339576682c46fa69459f76caa0bdf95f1f79f3ef3b42c2b5eb6744804`.

Независимая проба `runs/probe_teacher_cold_vertical.py` обнаружила ошибку
подготовки высоких начальных поз: `reset(target_angle=q)` задаёт запрошенную
цель, но сглаженная цель остаётся 75°. Все восемь noiseless teacher-опытов
с q=153°/155°/156°/157° и двумя профилями capacity покидают область 160°.
Например, старт 153° сначала опускается до 118.56°, затем разгоняется обратно.
Обычный физический подъём 75° → 155° проходит успешно: 155.00029° через 12 s.
Это проблема учебных начальных условий, а не доказательство невозможности
удержания 155° или недостаточной силы.

После последних параметров обучения `.venv/bin/python3 -m pytest -q`:
**69 passed, 2 warnings in 22.40 s**; `git diff --check` без замечаний.

### Достижимые начальные состояния возле вертикали

В `endurance_reachable` высокие начальные состояния получаются после 8 s
обычного подъёма teacher от 75°. При передаче управления сохраняются поза,
скорость, силы, утомление, очередь задержки и вся история. Это только подготовка
учебных эпизодов; runtime по-прежнему управляется одной сетью. Два номинальных
прохода к 155° продлены до 30 s; добавлены отпускания после 75° со свежими
и отдельно утомлёнными сгибателями. Всего 26 семейств.

Предварительная проверка `runs/probe_teacher_reachable_families.py` завершила
12/12 опытов: шесть подготовленных состояний при 1 и 1.05 kg, по 20 s
непрерывной физики. Максимум 157.8505°, конечная ошибка ≤0.00034°.
Дополнительное удержание 155° на 30 s с исходной capacity трицепсов 0.95
также прошло: максимум 155.00065°, итог 155.00029°. Эти результаты относятся
к teacher, а не к обученной сети.

```bash
.venv/bin/python3 -m myo.train --config configs/d1_train.json \
  --steps 0 --seed 57 --threads 1 --warm-start-steps 60000 \
  --command-imitation --imitation-rounds 3 \
  --resume-imitation runs/d1_actor_calibrated_control.zip \
  --imitation-curriculum endurance_reachable --imitation-vertical-noise 0 \
  --imitation-inactive-flexor-margin 0.01 --imitation-weak-weight 6 \
  --output runs/d1_actor_calibrated_reachable.zip
```

Запуск завершён: 60 тыс. новых переходов, 103 обучающих и 31 отдельный
validation-эпизод, 20 тыс. validation-переходов, без PPO. Выбраны эпохи
106/40/120; RMSE возбуждения 0.001620/0.005315/0.002104. Первый раунд
имел меньшую ошибку действий, но срывался в движении; после двух раундов
исправлений closed-loop score снизился с 18326.12 до 3.47.
Подготовка teacher не сорвалась ни разу; сбор на состояниях сети во втором
и третьем раундах завершил по пять эпизодов при выходе за учебные 160°.
Веса: `d716b86b5c02ca05204885c99f5f9b2fd6d0e7ef71c9738c51f52b72d1f6abd3`.
Суммарно цепочка шести этапов использовала 220 тыс. обучающих переходов;
валидация и физическая подготовка высоких поз в этот бюджет не входят.

Отдельный `--suite vertical` с обычным начальным углом 75° завершён:

| Цель | Угол в 30 s | RMS ошибки последних 5 s |
| --- | ---: | ---: |
| 140° | 139.3149° | 0.6858° |
| 150° | 149.8483° | 0.1482° |
| 155° | 155.5000° | 0.4661° |
| 175° | 230.9664° | 55.8287° |

Для 155° максимальный угол 155.8918°, возбуждение всех сгибателей в последнем
окне точно нулевое, среднее возбуждение трицепсов 15.588%. Для 175° максимум
364.8995°: опыт завершил заданное время без численного сбоя, но **удержание
провалено**, пути меняют роли за областью исходной анатомии. Само
`completed_duration=true` не означает успешного управления.

`--suite extension`: отпускание, принудительный ноль и пассивный отдых
заканчиваются на 17.16744° с нулевым возбуждением. Точная команда 15° с полным
триггером пока даёт 17.6229° и лишние 2.871% возбуждения сгибателей:
это остаточная ошибка actor, её нельзя объяснять только пассивным равновесием.
При отпускании proxy реакции нижнего ограничения 0 против 40.56 N·m
при forced zero, но пик силы сухожилия выше: 560.36 против 150.90 N.
Торможение уменьшает удар в этом опыте, не даёт общего доказательства защиты.

Полный прогон `.venv/bin/python3 -m myo.evaluate --controller
runs/d1_actor_calibrated_reachable.zip --suite endurance --output
runs/d1_actor_calibrated_reachable_endurance.json` завершился с кодом 0:

| Цель | Угол в 300 s | Capacity сгибателей в 300 s | Capacity после 120 s release | Возбуждение в конце отдыха |
| --- | ---: | ---: | ---: | ---: |
| 75° | 74.4835° | 77.30% | 95.78% | 0 |
| 125° | 124.7557° | 80.33% | 94.74% | 1.0004% |

Удержание работает, но при восстановлении до capacity около 0.95 после высокой
цели вновь появляется небольшой остаток возбуждения. Короткие release-проверки
с начальным профилем 0.8/1 этого не обнаружили. Wall time двух опытов
234.66/162.63 s; среднее время физического шага управления 10.19/7.09 ms,
policy 0.69/0.45 ms. Прогоны шли параллельно с другой оценкой, поэтому это
наблюдение текущей нагрузки ноутбука, а не отдельный аппаратный benchmark.

Пробы `runs/probe_rest_excitation_history.json` и `_02.json` не подтвердили,
что постоянная недавняя история возбуждения сама по себе достаточна для этого
остатка: после 8 s физического возбуждения сгибателей 0.01 или 0.02 сеть
сразу выдавала ноль и сохраняла его 12 s, как и teacher из идентичного состояния.
Конечный угол 17.16744°. Точная зависимость остатка от траектории не установлена;
короткий покой не заменяет длительный непрерывный опыт.

`--suite holds` даёт 74.3981°/119.7252°/124.6873° через 30 s; RMS ошибки
последнего окна 0.6042°/0.2828°/0.2966°. Отдельная цель 156° тоже удержана:
итог 156.063°, максимум 157.195°, RMS ошибки 0.543°; это одиночный опыт.
Однако `--suite commands` обнаружил существенную регрессию опускания:
при 75° → 45° рука сначала достигает 45.12°, затем сползает до 18.37°;
при 105° → 45° итог 18.23°. В обучении этот режим был представлен только
с capacity сгибателей 0.85. Поэтому даже этот checkpoint ещё нельзя считать
универсально исправленным демо.

Рендер реального движения `MUJOCO_GL=egl .venv/bin/python3
runs/render_final_actor_qa.py --controller runs/d1_actor_calibrated_reachable.zip
--output runs/final_actor_qa/actor-155.json` сохранил три просмотренных кадра.
На 147.276° активация сгибателей 5.075%, трицепсов практически нулевая;
на 154.205° — 0.097% и 9.216%; в 30 s на 155.50° — практически ноль и 18.516%.
Пути конечны, разворота их направления нет. Это фактические состояния сети,
не постановочные позы; GLFW-взаимодействие и кожа здесь не проверялись.

Дополнительная последовательность 75° → 120° → 60° → 125° → 155° обнаружила
границу, которой не видно в отдельном старте 75° → 155°: после команды
125° → 155° в 23 s получено 160.88° и 0.730 rad/s, затем выход в область
разворота путей. Последующие команды уже не дают проверки обычного движения;
отпускание выше 130° в этом диагностическом сценарии не выполнялось.
Отчёт — `runs/final_actor_qa/command-sequence.json`.

`runs/probe_teacher_125_to_155.json` воспроизводит ту же проблему у teacher:
после 8 s физического достижения 125° новая цель 155° за секунду приводит
к 160.616° и 0.630 rad/s. Такие демонстрации не используются как образец
успешного движения. Это известное ограничение резкой смены высокой цели
у существующего command/teacher, а не устраняемый увеличением imitation-budget
недостаток сети. Физический ограничитель для скрытия срыва не добавлялся.
Обратные teacher-переходы с сохранением истории проходят:
155° → 145° даёт 144.99947°, 145° → 60° — 59.99583° после 12 s;
максимумы 155.00064°/144.99947°. Результат —
`runs/probe_teacher_reverse_vertical.json`.

После добавления этой подготовки `.venv/bin/python3 -m pytest -q`:
**69 passed, 2 warnings in 35.90 s**. Физический fingerprint не изменился.
В логе текущего опыта поле `warmup_physical_steps` содержит число шагов
управления по 0.02 s, а не физических подшагов. В коде следующих запусков
оно переименовано в `warmup_control_steps` с явным `warmup_control_dt_s`.

### Сохранение освоенного поведения и переходы между целями

Выбор весов исправлен после подтверждённой регрессии: в `balanced` первый
раунд имел closed-loop score 27.66, второй — 2079.28, однако раньше сохранялся
второй. Теперь исходный checkpoint и лучшие кандидаты всех раундов сравниваются
на одном наборе сценариев. Метаданные сохраняют выбранный раунд, эпоху и score;
неудачный следующий fit не перезаписывает победителя.

`endurance_transitions` сохраняет предыдущие 26 семейств и добавляет шесть:
свежее опускание 75° → 45°, физически подготовленное 105° → 45°,
155° → 145°, 145° → 60° и покой с сохранёнными целями 125°/155° при capacity
сгибателей 0.90–0.98 и свежих разгибателях. Начальные высокие позы достигаются
обычным 8 s движением teacher. Полный цикл содержит 21 тыс. меток;
бюджет 22 тыс. на раунд и validation покрывает все 32 семейства.
Девять подготовительных проходов добавляют 3600 шагов управления вне бюджета
меток. Физика и преобразование входа сети не изменились.

```bash
.venv/bin/python3 -m myo.train --config configs/d1_train.json \
  --steps 0 --seed 67 --threads 1 --warm-start-steps 44000 \
  --command-imitation --imitation-rounds 2 \
  --resume-imitation runs/d1_actor_calibrated_reachable.zip \
  --imitation-curriculum endurance_transitions --imitation-vertical-noise 0 \
  --imitation-inactive-flexor-margin 0.01 --imitation-weak-weight 6 \
  --output runs/d1_actor_calibrated_transitions.zip
```

Этап завершён: 44 тыс. новых переходов, 70 обучающих и 34 отдельных
validation-эпизода, 22 тыс. validation-переходов. Выбраны эпохи 80/80;
RMSE возбуждения 0.001974/0.002245. Score исходного checkpoint 10060.36,
после раундов 5097.76/3.00616. В первом раунде свежие опускания уже исправились,
но высокая цель колебалась; второй раунд исправлений на состояниях сети
восстановил её удержание. Все 32 семейства представлены в каждом раунде.
Сбор на actor завершил два эпизода при выходе за учебные 160°;
физических срывов и неудачных подготовительных проходов не было.
Финальный SHA:
`816cad5d63ba518d0f4e4771bc199e13897de2ecc10100739845080c9ef18399`.
Суммарно семь этапов использовали 264 тыс. обучающих переходов. Это imitation
с исправлениями на собственных состояниях, без последующих обновлений PPO.

После исправления отбора `.venv/bin/python3 -m pytest -q`:
**70 passed, 2 warnings in 26.37 s**. Первый запуск в ограниченном sandbox
завершился как 69 passed/1 failed из-за запрета локального UDP-сокета;
повтор с разрешённым loopback прошёл без изменения теста.
После добавления уже согласованного обратного маршрута в официальный
`--suite vertical`: `.venv/bin/python3 -m pytest -q tests/test_evaluate.py` —
**2 passed in 0.69 s**. Это сценарий, также используемый для выбора весов,
а не независимый holdout.

### Итоговые проверки сохранённого actor

Проверялся `runs/d1_actor_calibrated_transitions.zip` с указанным выше SHA,
без дообучения между сценариями. Физический fingerprint всех прогонов:
`23f8a1313177949964d42521e2fa721a0500a5cc733d92a551c77ed4ff143fa8`.
Фактические команды:

```bash
.venv/bin/python3 runs/evaluate_transitions_suites.py
.venv/bin/python3 -m myo.evaluate --controller runs/d1_actor_calibrated_transitions.zip \
  --suite endurance --output runs/d1_actor_calibrated_transitions_endurance.json
.venv/bin/python3 runs/evaluate_transitions_jump.py
MUJOCO_GL=egl .venv/bin/python3 runs/render_final_actor_qa.py \
  --controller runs/d1_actor_calibrated_transitions.zip \
  --output runs/transitions_actor_qa/actor-155.json
```

Первый локальный helper последовательно вызывает официальный `myo.evaluate`
с `--suite vertical`, `extension`, `holds`, `commands`, затем дополнительный
30 s опыт с целью 156°. Отчёты:
`runs/d1_actor_calibrated_transitions_{vertical,extension,holds,commands,156}.json`.
Все процессы завершились с кодом 0; это само по себе не критерий успеха движения.

При массе 1 kg, свежих мышцах, старте с 75° и новой цели с 2 s:

| Цель | Угол в 30 s | RMS ошибки последних 5 s |
| --- | ---: | ---: |
| 75° | 74.5251° | 0.4801° |
| 120° | 119.8423° | 0.1439° |
| 125° | 125.4142° | 0.4127° |
| 140° | 139.7798° | 0.2381° |
| 150° | 149.9282° | 0.0701° |
| 155° | 155.2041° | 0.1363° |
| 156° | 155.175° | 0.6473° |

При цели 155° максимум 155.4630°, стандартное отклонение угла последних
5 s — 0.0611°. Возбуждения всех сгибателей в этом окне равны нулю,
среднее возбуждение трицепсов 12.66%, средняя активация 12.79%.
Обратный маршрут 75° → 155° → 145° → 60° завершился на 60.1028°,
RMS последнего удержания 0.1008°. Этот маршрут участвовал в выборе весов;
его успешное повторение не является независимой оценкой обобщения.

Команда удерживать 15° с полным триггером даёт 14.999934°: сгибатели выключены,
возбуждение трицепсов 6.2314%. Оно нужно против пассивного сопротивления модели.
Отпускание, принудительное обнуление и пассивный покой дают одинаковые 17.16744°
с нулевыми возбуждениями всех шести мышц в последнем окне. Runtime не маскирует
выход actor при trigger=0; транспортная задержка и активация сохранены.

В `commands` свежее опускание к 45° завершается на 44.6599°, сценарий
подъём–удержание–опускание — на 44.6006°. При trigger=0.1 цель 105° даёт
103.7504° за 6 s. Обычное отпускание и отпускание с исходной capacity 0.7
заканчиваются в пассивном покое с нулевыми возбуждениями. Однако до отпускания
в последнем сценарии RMS tracking составляет 20.3189°: уверенное управление
при одновременном сильном снижении capacity всех мышц не подтверждено.

Два непрерывных опыта `endurance` длятся по 420 s: команда удержания до 300 s,
затем 120 s trigger=0. Масса остаётся 1 kg, сбросов состояния и восстановления
в обход закона утомления нет.

| Цель | Угол перед отпусканием, 300 s | Capacity сгибателей, 300 s | Capacity сгибателей, 420 s |
| --- | ---: | ---: | ---: |
| 75° | 74.6471° | 77.2980% | 95.7727% |
| 125° | 125.9469° | 80.4010% | 96.3499% |

Оба опыта завершили всю длительность без физических/численных сбоев.
В 420 s угол 17.16744°, все возбуждения последнего окна равны нулю.
Восстановление — результат текущего capacity proxy; эти числа не устанавливают
время восстановления человека. Wall time 158.13/134.63 s, среднее время
шага управления физикой 6.918/5.884 ms, сети 0.413/0.358 ms; одновременно
выполнялись другие локальные проверки, поэтому это не изолированный benchmark.

EGL сохранил три реально просмотренных кадра финальных весов:
при 147.55° активация сгибателей 4.56%, трицепсов практически нулевая;
при 154.07° — 0.053%/0.657%; в 30 s при 155.20° — практически ноль/14.26%.
Все пути конечны, разворота плеч в этих кадрах нет. Это исходные кости и
процедурные ткани; проверка не включает кожу, UE или ввод в настоящем GLFW-окне.
Небольшое перекрытие подписи груза/кисти на высоких углах сохраняется.

Оставшиеся неудачные опыты записаны явно:

- При цели 175° максимум 365.26°, итог 233.47°. Отсутствие numerical failure
  не означает успешное управление: модель попадает за область исходных путей.
- Прямой переход 125° → 155° после 8 s удержания повторно проверен на итоговых
  весах (`runs/transitions_125_to155.json`): в 9 s угол 176.75°, скорость
  3.761 rad/s; максимум 365.37°, итог 237.07°, RMS последнего окна 81.99°.
  Аналогичный срыв teacher описан выше. Для таких переходов текущий контроллер
  не готов; скрытого верхнего ограничителя или спасательного сброса не добавлено.

## Исправление нейросетевого демо — 2026-11-01

Найдена ошибка начальных условий: `demo --controller runs/d1_actor_initial.zip`
с seed 0 использовало учебную рандомизацию checkpoint (`task_mode=curriculum`).
Получались **1.4554 kg и capacity=0.78094**, хотя обычное interactive-демо
начиналось с 1 kg и capacity=1. Теперь только `demo` при старте и явном **R**
передаёт номинальную массу конфигурации и capacity=1; обучение и headless-rollout
сохраняют рандомизацию. Во время движения состояние не сбрасывается.

Отдельно воспроизведена ошибка самого старого actor на равных условиях:
1 kg, capacity=1, начальный угол 75°, полный триггер, новая цель с 2 s,
непрерывный прогон 30 s. Физика, утомление, задержка и мягкий ограничитель
совпадают с checkpoint; `dynamics_sha256` не менялся.

```bash
.venv/bin/python3 runs/policy_repair_metrics.py --controller runs/d1_actor_initial.zip --output runs/policy_old_metrics.json
.venv/bin/python3 runs/policy_repair_metrics.py --controller heuristic --output runs/policy_teacher_metrics.json
.venv/bin/python3 runs/policy_repair_sequence.py --controller runs/d1_actor_initial.zip --output runs/policy_old_sequence.json
.venv/bin/python3 runs/policy_repair_sequence.py --controller heuristic --output runs/policy_teacher_sequence.json
```

| Цель | Старый actor: угол в 30 s | Heuristic: угол в 30 s |
| --- | ---: | ---: |
| 75° | 57.90° | 74.95° |
| 120° | 102.52° | 119.94° |
| 125° | 105.28° | 124.87° |

У старого actor при удержании 75° к концу осталось 70.19% capacity сгибателей,
у heuristic — 73.62%. При цели 125° сеть достигла максимум 117.73° и затем
сползала при недостаточном возбуждении. В старых условиях окна (1.4554 kg,
capacity=0.78094) максимумы для целей 120°/125° — только 102.56°/105.24°.
Это не подтверждает гипотезу, что вся ошибка объясняется утомлением.

В отдельной последовательности 75° → 120° → 60° → 125° (34 s) средние углы
старого actor в последних 3 s каждой команды — 64.08°/110.26°/50.28°/104.18°.
После удаления линейного дрейфа стандартное отклонение угла 0.001–0.005°;
смен знака скорости нет. **Устойчивое угловое дрожание этим опытом не
воспроизведено**; проверка настоящего GLFW-окна по-прежнему не выполнена.

`.venv/bin/python3 -m pytest -q`: **62 passed, 1 warning in 22.78 s**.
Запуск разрешал локальный loopback для UDP-теста. Добавлены проверки номинальных
условий при старте/reset демо и переноса нормализации вместе с PPO ZIP; изменение
кода преобразования признаков отвергается по fingerprint. Прежние checkpoints
без преобразования признаков сохраняют совместимость.

### Исправленный imitation actor

Фактически выполнено локально на CPU:

```bash
.venv/bin/python3 -m myo.train --config configs/d1_train.json \
  --steps 0 --seed 7 --warm-start-steps 45000 \
  --command-imitation --imitation-rounds 3 --log-std-init -4 \
  --output runs/d1_actor_corrected.zip > runs/d1_actor_corrected.log
.venv/bin/python3 -m myo.evaluate --controller runs/d1_actor_corrected.zip \
  --output runs/d1_actor_corrected_evaluation.json
.venv/bin/python3 -m myo.evaluate --suite holds --controller runs/d1_actor_initial.zip \
  --output runs/d1_actor_initial_holds.json
.venv/bin/python3 -m myo.evaluate --suite holds --controller runs/d1_actor_corrected.zip \
  --output runs/d1_actor_corrected_holds.json
.venv/bin/python3 runs/policy_repair_metrics.py --controller runs/d1_actor_corrected.zip \
  --output runs/policy_corrected_metrics.json
.venv/bin/python3 runs/policy_repair_sequence.py --controller runs/d1_actor_corrected.zip \
  --output runs/policy_corrected_sequence.json
.venv/bin/python3 runs/policy_repair_metrics.py --controller runs/d1_actor_corrected.zip \
  --targets 45 90 110 130 --output runs/policy_corrected_extra_metrics.json
```

MLP 128×128 получает все прежние причинные наблюдения с историей. Нормализация
вычислена только по первому тренировочному набору и затем заморожена; buffers
переезжают вместе с ZIP. Три раунда по 15 тыс. переходов, всего 60 эпизодов;
вероятность выполнения teacher-action 1.0 → 0.3 → 0.0. Во всех раундах label —
heuristic на **текущем** состоянии; небольшие возмущения teacher-action при
сборе данных не попадают в label. Оптимизация — 180 эпох на раунд с выбором
по отдельным validation-эпизодам. В runtime остаётся только один actor без
подмешивания heuristic, прямого момента или изменения мышечной механики.

Validation: 4000 переходов, 5 отдельных эпизодов, seed 110007; RMSE возбуждений
по раундам 0.2163 → 0.1965 → 0.1705. Это небольшой и неравномерный набор:
55% переходов относятся к удержаниям 128–130°. Поэтому само уменьшение этой
ошибки не считается доказательством управления; ниже отдельные замкнутые опыты.

| Цель, непрерывно 30 s | Старый actor: конечный угол | Новый actor: конечный угол | Новый: RMSE последних 5 s |
| --- | ---: | ---: | ---: |
| 75° | 57.90° | 74.18° | 0.99° |
| 120° | 102.52° | 115.41° | 4.62° |
| 125° | 105.28° | 123.75° | 1.02° |

При цели 125° максимум нового actor **125.54°**. При удержании 75° конечная
capacity сгибателей **73.38%**, близко к heuristic **73.62%**. Модель утомления
не ослаблялась; это инженерный proxy, а не физиологически откалиброванный темп.
Для дополнительных целей 45°/90°/110° конечные углы 43.25°/86.13°/105.12°:
остаточная ошибка обратной связи заметна и в середине диапазона.
При цели 130° достигнут максимум 129.57°, затем угол снижается до 124.81°
при полном возбуждении сгибателей и capacity 0.280. Здесь сохраняется
физическое ограничение утомлённой модели, описанное ниже для heuristic.

Три основных удержания дополнительно воспроизведены через сохранённый в коде
`myo.evaluate --suite holds`: все шесть сравнительных прогонов старого и нового
actor завершили 30 s без срыва и подтвердили конечные углы таблицы. Режим
продолжает учебный time limit без сброса физического состояния, как `demo`;
обычные шесть сценариев оценки сохраняют прежнее ограничение эпизода.
Итоговый `.venv/bin/python3 -m pytest -q`: **63 passed, 1 warning in 14.12 s**;
предупреждение SB3 о диапазоне действий прежнее. `git diff --check` — без ошибок.

При смене 75° → 120° → 60° → 125° новый actor имеет средние углы в последних
3 s удержания 73.08°/118.68°/56.45°/124.24°. Остаточные изменения — преимущественно
медленный дрейф; после его удаления STD угла не превышает 0.012°.
В дополнительных тяжёлых условиях (1.455 kg, начальная capacity 0.781, 20 s)
цели 110°/125° заканчиваются на 108.63°/122.45°, RMSE последних 5 s 0.69°/1.99°.
Отчёты: `runs/review_corrected_heavyweak.json`, `runs/review_teacher_heavyweak.json`.
Все перечисленные прогоны завершились без численного/физического срыва.

Стандартный подъём–удержание–опускание: RMSE **5.35°** вместо прежних **9.73°**;
управляемое опускание — **4.79°**. Все шесть сценариев завершены; состояния
перед ветвлением controlled lower/release/forced zero совпадают.
После release скорость движения к пределу **1.48 rad/s** против **4.30 rad/s**
при forced zero; реакция жёсткого ограничителя **0 против 40.29 Nm**.
При этом пик силы сухожилия **823 N против 166 N**: снижение скорости удара
не означает улучшения всех нагрузочных метрик. После release tracking-ошибка
не учитывается, потому что нулевой триггер снимает задачу удержания угла.

Это улучшенная imitation-policy, **не успешная PPO-policy и не гарантия защиты**.
Компенсация утомления остаётся неточной, особенно при малой capacity и высоких
углах. Новые веса сохранены отдельно; прежние не перезаписывались. Настоящее
GLFW-окно, субъективная плавность и произвольные команды пользователя здесь
не проверены. `runs/` содержит локальные артефакты и не включается в репозиторий.

## Диагностика сгибания около 125° — 2026-11-01

По наблюдению пользователя выполнены headless-сравнения при цели 130°, старте
75°, грузе 1 kg и исходной capacity=1. Производственные настройки и веса не
менялись; варианты без утомления/пассивного ограничителя созданы только для
сравнения в памяти. Лимит эпизода продолжен так же, как в интерактивном окне.

```bash
.venv/bin/python3 runs/upper_angle_qa/heuristic.py > runs/upper_angle_qa/heuristic.log
```

| Heuristic, длительность 30 s | Максимальный угол | Угол в 30 s | Capacity сгибателей в 30 s |
| --- | ---: | ---: | ---: |
| Обычные настройки, trigger=1 | 129.47° | 124.45° | 0.267 |
| Без утомления, trigger=1 | 130.00° | 130.00° | 1.000 |
| Без мягкого ограничителя, trigger=1 | 129.93° | 129.93° | 0.562 |
| Обычные настройки, trigger=0.5 | 128.01° | 122.94° | 0.457 |

В обычном варианте на 25 s угол **125.62°**, возбуждение сгибателей **1.0**,
capacity **0.311**. Четыре прогона завершились без физического termination.
Подробности сил/моментов/длин каждые 5 s: `runs/upper_angle_qa/heuristic.json`.

Мягкая пружина ограничителя начинается при 120°; её сопротивление при нулевой
скорости равно **1.52 Nm** на 125° и **6.09 Nm** на 130°. Это выбранный
инженерный параметр, не откалиброванное сопротивление локтя человека.
Гравитационный момент на этих углах уменьшается: **2.17 → 1.67 Nm**.
Сгибатели при этом укорачиваются и теряют силовую эффективность; растянутые
разгибатели создают пассивный разгибающий момент.

Отдельно решено статическое мышечно-сухожильное равновесие в фиксированных позах
с активацией сгибателей 1 и разгибателей 0. Ни фиксация позы, ни мгновенная
активация из этого диагностического расчёта не перенесены в runtime:

| Угол | Требуемый мышечный момент, Nm | Доступный при capacity=1 | При capacity=0.3 |
| --- | ---: | ---: | ---: |
| 120° | 2.655 | 21.977 | 5.657 |
| 125° | 3.693 | 17.318 | 4.169 |
| 130° | 7.761 | 12.656 | 2.718 |

Доступный момент учитывает пассивное противодействие разгибателей.
Это статические значения данного приближения, не обещание достижимости позы
при задержках, движении и накоплении утомления. Файл: `runs/upper_angle_static.json`.

`d1_actor_initial.zip` в отдельном 20 s опыте с той же командой достигает
**120.61° на 2 s**, при capacity **0.970** и возбуждении сгибателей около
**0.206**; к 20 s угол **112.83°**, capacity **0.755**. Здесь недостаточная
команда actor видна до значимого утомления. Файл: `runs/upper_angle_actor.json`.
Предел 125° не зашит в код: верхняя граница диапазона 130°, а конкретная
недостижимая поза зависит от настроек сопротивления, состояния и контроллера.

## Непрерывное демо и раздельные ткани — 2026-11-01

Исправлен автоматический reset окна по лимиту обучающего эпизода: раньше он
происходил каждые 20 s для interactive-конфига или 10 s для текущих весов.
`demo` и `rollout --viewer` продолжают отсчёт времени, мышечное состояние,
утомление, задержанные команды и историю. Конфигурация checkpoint не меняется.
Физический guard останавливает окно до **R**, **Space** его не обходит.
Headless-rollout, обучение и оценка сохраняют прежние границы эпизодов.

В режиме **1 / tissues** одновременно показаны активация брюшка и натяжение
сухожилия. Раздельные длины волокна и сухожилия определяют границы тканей;
сокращение утолщает брюшко, растяжение утончает. Для сухожилия утончение действует
сверх slack length; нулевая длина не создаёт бесконечного радиуса. Геометрия
сохраняет нативные огибания MuJoCo. Это процедурная диагностическая форма,
не BodyParts3D-поверхности, Flesh или кожа. Суммарное сухожилие поделено между
концами поровну — явное визуальное допущение.

Фактически выполнено:

```bash
.venv/bin/python3 -m pytest -q tests/test_demo.py -k 'cli or horizon or physical_failure'
.venv/bin/python3 -m pytest -q tests/test_demo.py
.venv/bin/python3 -m pytest -q
MUJOCO_GL=egl .venv/bin/python3 runs/render_tissue_qa.py
git diff --check
```

Проверка цикла: **6 passed**; целевая проверка окна/геометрии: **14 passed**;
итоговый полный набор после исправления нулевой длины: **60 passed, 1 warning
in 13.58 s**. Прежнее предупреждение SB3 о диапазоне действий `[0,1]`.
Для полного набора разрешены локальные UDP-сокеты. `git diff --check` чист.

Дополнительно через настоящий CLI/физику MyoArm, с подменённым окном и без
ожидания wall clock, выполнены **42 s heuristic** и **22 s imitation actor**.
В обоих случаях один эпизод, ноль физических срывов, монотонное время пакетов;
пройдены по два прежних лимита без reset. Артефакты:
`runs/continuous_demo_qa/`. Это проверка цикла и загрузки прежних весов,
не проверка настоящих событий GLFW или качества длительного управления.

Через EGL отрисованы девять кадров: 15°, 45°, 75°, 105°, 129°, сокращение
при фиксированных 75° и три дополнительных цветовых режима. Проверены крайние
позы и различие расслабленной/напряжённой ткани, читаемость двух легенд;
камера сдвинута, чтобы уменьшить перекрытие кисти подсказками. Кадры и измерения:
`runs/tissue_viewer_qa/`. При фиксированных костях и возбуждении 0.8 на 300
мышечных подшагах у BIClong длина волокна изменилась **128.01 → 119.73 mm**,
сухожилия **277.04 → 285.31 mm**, радиус брюшка **10.07 → 10.41 mm**.
Это изометрический проверочный стенд мышцы; прикладывать скрытый момент
фиксации в интерактивную физику он не добавляет. Рендер не меняет физическое
состояние, что проверяется снимками до/после.

Настоящее GLFW-окно в текущем окружении без дисплея не запускалось. Физический
fingerprint не менялся; ранее сохранённый imitation actor совместим.
Две постоянные цветовые полосы кешируются, чтобы не вычислять их каждый кадр.

## Linux-этап D1 — 2026-11-01

Реализованы и проверены MyoArm с одним подвижным локтем, шесть внешних
мышечно-сухожильных приводов, команды угла/триггера, 200 ms истории,
сценарная оценка, MuJoCo-изображение и Python-канал локальных команд.
UE, Flesh, BodyParts3D-поверхности мышц, подкожный объём и кожа не создавались.
Кожа остаётся обязательным критерием полного D1.

Окружение: Linux, Python **3.12.16**, MuJoCo **3.17.0**, Gymnasium **1.4.0**,
NumPy **2.5.4**, PyTorch **2.15.0+cpu**, Stable-Baselines3 **2.9.2**,
pytest **9.2.0**. Полный проверенный список — `requirements-verified.txt`.
Шаг физики 1 ms, управление 20 ms, моторная задержка 40 ms.
Измерения относятся к текущему Linux-окружению; это не результат Windows/UE.

### Установка и импорт

Фактически выполнено:

```bash
UV_CACHE_DIR=/tmp/myo-uv-cache PATH="$HOME/.local/bin:$PATH" bash scripts/setup.sh
UV_CACHE_DIR=/tmp/myo-uv-cache "$HOME/.local/bin/uv" pip install \
  --python .venv/bin/python3 torch --index-url https://download.pytorch.org/whl/cpu
UV_CACHE_DIR=/tmp/myo-uv-cache "$HOME/.local/bin/uv" pip install \
  --python .venv/bin/python3 --editable '.[train]'
.venv/bin/python3 scripts/prepare_myoarm.py
```

`setup.sh` завершился успешно, `pip check` не нашёл несовместимостей.
Для установки потребовалось разрешение sandbox на сеть; для cache использован
записываемый `/tmp`. Облачные сервисы обучения не подключались.
MyoArm закреплена на `93b0ca8f4ec90c9899ee7f05fee561e9911da91b`: 39 проверенных
исходных файлов, 33 mesh assets, 1 DOF, 0 встроенных actuators, 6 пространственных
путей. Исходные и производные файлы находятся в игнорируемом `data/myoarm`.
Происхождение и ограничения — [myoarm.md](myoarm.md).

### Автоматические проверки

```bash
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m pytest -q tests/test_demo.py tests/test_commands.py
git diff --check
```

Итоговый полный набор после добавления imitation initialization и CLI:
**55 passed, 1 warning in 21.72 s**. Предупреждение SB3 рекомендует
симметричный action space; наш `[0,1]` сохраняет явную семантику возбуждения.
После исправления учёта текущего trigger в CLI и выбора ручного режима для
внешнего ввода целевая повторная проверка: **13 passed in 1.50 s**.
`git diff --check` — без замечаний. Полный набор запускался с разрешением
локальных сокетов: без него UDP-тест блокируется sandbox до отправки пакета.

### Механика и изображение

Проверены равенство `r=-dL/dq`, совпадение концов отображаемых путей с физическими
sites, отсутствие нативной мышечной/сухожильной силы, крепление груза к кисти,
независимость отрисовки от состояния физики. На дополнительной сетке из 101 позы
максимальное расхождение плеча и конечной разности — **1.22e-10 m**.

Два 10 s опыта toy-геометрии при шагах 1/0.5/0.25 ms показывают сходимость первого
порядка. Максимальное расхождение приложенной силы на паре 1/0.5 ms — **0.141 N**;
силы снимка конечной позы — **1.430 N**, при следующем делении шага — **0.719 N**.
Снимок и приложенная сила намеренно различаются: повторного шага ткани нет.
При прогоне всех шести мышц по реальным путям MyoArm, 4 s, 15–130°, 0.5 Hz,
с возбуждением 0 и 1 максимальная невязка решения **1.17e-7 N**, не более
6 итераций решателя. Это ограниченные численные проверки, не анатомическая
валидация или доказательство устойчивости для любых параметров.

Через `MUJOCO_GL=egl` и `myo.demo.render_frame()` отрисованы и просмотрены
45°, 75°, 105°, 129° при 960×720, режимы активации, силы и strain. Кадры:
`runs/viewer_qa/`. Камера исправлена против перекрытия кисти числовой панелью.
На изображении кости и пути мышц; объёмные мышечные брюшки и кожа отсутствуют.
События клавиатурной очереди проверены тестом; настоящее GLFW-окно с мышью
в этом окружении не проверялось.

Для самой MyoArm дополнительно выполнен `python3 runs/qa_timestep.py`:
6 s удержания/отпускания heuristic при шагах 1 и 0.5 ms с неизменными периодом
управления 20 ms и задержкой 40 ms. Максимальные расхождения: угол **0.00799 rad**,
скорость **0.0793 rad/s**, активация **0.0200**, приложенная сила **16.95 N**.
Контроллер с пороговым торможением выбирает разные команды на немного различных
траекториях; это измерение чувствительности замкнутого контура, не утверждение
об эквивалентности бэкендов. Результат: `runs/timestep_qa.json`.

### Сценарии и обмен

```bash
.venv/bin/python3 -m myo.evaluate --controller heuristic --seed 101 \
  --output runs/d1_heuristic_evaluation.json
.venv/bin/python3 -m myo.evaluate --controller zero --seed 101 \
  --output runs/d1_zero_evaluation.json
.venv/bin/python3 runs/qa_loopback.py
```

Оба набора из шести сценариев завершены без численного сбоя или выхода за
аварийные границы. Ветви controlled_lower/release/forced_zero имеют одинаковый
hash состояния перед разделением **внутри каждого контроллера**.
Heuristic в подъёме–удержании–опускании: **RMSE 5.58°**, конечный угол **44.93°**.
Слабый триггер 0.1 не поднимает груз 1 kg к 105°: конечный угол **17.79°**.

Сравнение после 2 s одинакового удержания heuristic:

| Метрика после отпускания | Heuristic с торможением | Принудительное нулевое возбуждение |
| --- | ---: | ---: |
| Пик скорости к нижнему пределу | 1.90 rad/s | 4.34 rad/s |
| Пик реакции численного joint limit | 0 Nm | 40.85 Nm |
| Пик отдельной сухожильной силы | 857.49 N | 164.75 N |

Торможение уменьшает удар, но увеличивает натяжение. Это результат эвристики,
не доказательство обученной защиты; нагрузки названы инженерными прокси.
У `zero` подъём–удержание–опускание даёт RMSE **63.78°**.

`runs/qa_loopback.py` — локальный проверочный сценарий CLI: получен 51 пакет
V2 за 1 s симуляции, команда trigger=0.7 принята, устаревший sequence отвергнут,
после timeout trigger стал 0; масса сохранилась, физического срыва не было.
Результат — `runs/loopback_qa.json`. Сокеты использовали только `127.0.0.1`;
для проверки понадобилось разрешение sandbox на локальные сокеты.

### Обучение: проверка поведения, включая неудачные опыты

Фактически выполнены следующие локальные CPU-опыты без predictor:

```bash
.venv/bin/python3 -m myo.train --config configs/d1_train.json \
  --steps 100000 --seed 0 --output runs/d1_ppo.zip
.venv/bin/python3 -m myo.train --config configs/d1_train.json \
  --steps 10000 --seed 1 --warm-start-steps 10000 --log-std-init -2 \
  --output runs/d1_ppo_warm.zip
.venv/bin/python3 -m myo.train --config configs/d1_train.json \
  --steps 0 --seed 1 --warm-start-steps 10000 --log-std-init -2 \
  --output runs/d1_actor_initial.zip
.venv/bin/python3 -m myo.train --config configs/d1_train.json \
  --steps 5000 --seed 1 --warm-start-steps 10000 --log-std-init -4 \
  --learning-rate 0.0001 --target-kl 0.01 --output runs/d1_ppo_low_noise.zip
.venv/bin/python3 -m myo.evaluate --controller runs/d1_ppo.zip \
  --seed 101 --output runs/d1_ppo_evaluation.json
.venv/bin/python3 -m myo.evaluate --controller runs/d1_ppo_warm.zip \
  --seed 101 --output runs/d1_ppo_warm_evaluation.json
.venv/bin/python3 -m myo.evaluate --controller runs/d1_actor_initial.zip \
  --seed 101 --output runs/d1_actor_initial_evaluation.json
.venv/bin/python3 -m myo.evaluate --controller runs/d1_ppo_low_noise.zip \
  --seed 101 --output runs/d1_ppo_low_noise_evaluation.json
```

| Контроллер | Демонстрации / PPO transitions | RMSE подъём–удержание–опускание | Итог |
| --- | ---: | ---: | --- |
| Heuristic | — | 5.58° | Диагностический ориентир |
| PPO с нуля | 0 / 100096 | 63.75° | Почти нулевые возбуждения, груз внизу |
| Imitation → PPO | 10000 / 10112 | 63.12° | Дообучение разрушило полезную инициализацию |
| Только imitation | 10000 / 0 | 9.73° | Полезное, но неточное выполнение команд |
| Imitation → PPO, меньшие шум и learning rate | 10000 / 5120 | 46.37° | Избыточное сгибание, команда release не выполняется |

Обучение с нуля завершилось примерно за 629 s, но **не дало рабочей policy**.
Низкие нагрузки после release у провалившихся PPO не доказывают торможения:
они не удержали исходный угол ещё до отпускания.
Последний короткий опыт с `std≈0.0183`, ограничением KL и меньшим learning rate
тоже ухудшил actor: после release рука осталась на **123.39°**. Он не выбран для
демо. Изменение этих параметров само по себе не исправило обучение; необходима
дальнейшая работа над PPO и reward, а не увеличение бюджета вслепую.

Imitation обучает тот же MLP actor `[64,64]` на причинных демонстрациях
диагностического heuristic. Seed демонстраций 10001; 18 эпизодов для обучения,
2 целиком отделённых эпизода для выбора весов; 100 epochs. RMSE возбуждения
на этих validation-эпизодах **0.01465**. Это другая метрика, не ошибка движения.
В runtime действует только actor; нет переключения на teacher, отдельной
защитной сети или недоступных будущих команд. Результат получен имитацией
инженерной эвристики, а не самостоятельным открытием защитного поведения PPO.

Для imitation-only все шесть стандартных сценариев завершились без численного
сбоя или аварийного выхода; состояния перед разделением ветвей идентичны.
После подъёма и опускания итоговый угол **36.59°** при цели 45° — заметная
неточность остаётся. Слабый триггер 0.1 не поднял груз, итог **17.95°**.
Сравнение release после одинаковых 2 s удержания actor:

| Метрика после отпускания | Actor | Принудительный ноль после того же actor |
| --- | ---: | ---: |
| Пик скорости к нижнему пределу | 1.23 rad/s | 4.10 rad/s |
| Пик реакции численного joint limit | 0 Nm | 37.53 Nm |
| Пик отдельной сухожильной силы | 675.14 N | 180.22 N |
| Среднее превышение trigger по мышцам и времени | 0.0281 | 0 |

Торможение здесь уменьшило скорость/удар, но увеличило натяжение; нулевая реакция
численного ограничителя не означает отсутствия пассивного сопротивления.
Этого опыта недостаточно для общей гарантии защиты или принятия полного D1.

Дополнительно выполнен 10 s rollout сохранённого actor в его curriculum:

```bash
.venv/bin/python3 -m myo rollout --controller runs/d1_actor_initial.zip \
  --seconds 10 --output runs/d1_actor_rollout.jsonl
```

RMSE **10.51°** на 405 кадрах с trigger > 0, физических срывов 0.
500 control steps: среднее `env.step` **8.31 ms**, p95 **8.97 ms**;
среднее исполнение policy **0.418 ms**, p95 **0.593 ms**. Время симуляции 10 s,
wall time **4.84 s** с записью JSONL. Параллельно выполнялись другие CPU-проверки;
это измерение данного запуска без окна, не FPS графики или UE.
Физический fingerprint всех этих весов:
`844ab19ae6e64fba812243c727d525dcca01de30d70ec559afe26016a38a938c`.

### Отложенные условия для imitation actor

После фиксации `d1_actor_initial.zip` отдельно выполнено:

```bash
.venv/bin/python3 runs/d1_holdout_audit.py > runs/d1_holdout_audit.log
```

Четыре сценария выбраны до просмотра их результатов. Они не использовались
для подбора гиперпараметров actor; содержат другие условия, а не только другой
seed стандартной фиксированной оценки. Начальный угол 75°, trigger=1;
неуказанные масса/capacity/скорость равны 1 kg / 1 / 0 rad/s.

| Сценарий, seed | Условия и команды от начала опыта |
| --- | --- |
| `light_lift_release`, 4201 | 0.5 kg; 105° в 0.8 s, release в 2.4 s; длительность 6 s |
| `heavy_fatigued_release`, 4202 | 2 kg, capacity=0.7; release в 2.4 s; 6 s |
| `moving_early_release`, 4203 | Начальная скорость −1 rad/s; release в 0.4 s; 4 s |
| `reversal_release_repress`, 4204 | 105° в 0.6 s, 45° в 1.6 s, release в 2.3 s, 90°/trigger=1 в 3.7 s; 7 s |

Каждый выполнен с actor, heuristic и принудительным нулём после того же actor.
Все 12 прогонов дошли до заданной длительности без численного сбоя и срабатывания
грубого аварийного guard. Начальные состояния совпадают; во всех четырёх парах
actor/forced-zero совпадают также состояния перед отпусканием.

| Метрика по четырём сценариям | Actor | Heuristic | Ноль после actor |
| --- | ---: | ---: | ---: |
| Tracking RMSE при trigger > 0 | 12.08° | 7.88° | 38.18° |
| Максимальная скорость разгибания после release, на control frames | 4.27 rad/s | 4.39 rad/s | 5.09 rad/s |
| Пик реакции joint limit после release | 38.80 Nm | 31.19 Nm | 117.68 Nm |
| Пик отдельной сухожильной силы после release | 726.55 N | 871.45 N | 250.20 N |
| Среднее возбуждение после release | 0.0368 | 0.0377 | 0 |

Метрики release используют только кадры с trigger=0; пики сил — физические
подшаги. Forced-zero остаётся выключенным после повторного нажатия, поэтому
его общий tracking RMSE включает намеренный отказ от этой команды.
Heuristic и actor имеют разные траектории до release; их ветви напрямую
не являются опытом из одного физического состояния.

**Тяжёлый груз со сниженной capacity всё же достигает ограничителя:** у actor
минимум угла 14.36° при номинальном нижнем пределе 15° и ненулевая реакция joint
limit. Отсутствие аварийного termination не означает успешной защиты.
Результаты: `runs/d1_holdout_audit.json`, трассы: `runs/d1_holdout_traces.jsonl`.
Имитация полезна для исследования, но поведенческая приёмка полного D1 не закрыта.

---

# Историческая проверка V0

Дата: **2026-10-29**. Проверки выполнены локально в headless Linux-окружении.
Это отчёт о реализации и ограниченных экспериментах, не валидация биомеханики
человека и не измерение будущей UE-сцены.

## Окружение

- Python 3.12.14.
- MuJoCo 3.17.0, Gymnasium 1.4.0, NumPy 2.5.4.
- PyTorch 2.15.0+cpu, Stable-Baselines3 2.9.2, pytest 9.2.0.
- Полный список установленного окружения: [requirements-verified.txt](../requirements-verified.txt).
- Шаг физики 1 ms, период управления 20 ms, моторная задержка 40 ms.
- Собственная модель: 1 DOF, четыре мышцы, nominal payload 1 kg.

GPU, Unreal Engine, графический дисплей и анатомические datasets для этих
проверок не использовались. Данные и веса опытов записывались в игнорируемый
`runs/`; в репозитории остаются код, конфигурации, тесты и этот отчёт.

## Автоматические тесты

```bash
OPENBLAS_NUM_THREADS=1 python3 -m pytest -q
```

**76 passed**, одно предупреждение SB3, время данного запуска 7.85 s.
Предупреждение рекомендует симметричный диапазон actions; в V0 намеренно
используется физический контракт excitation `[0,1]`.

Проверены:

- точная задержка и её отличие от фильтра активации;
- натяжение, slack, равновесие, изометрическое укорочение волокна и энергия;
- fatigue/recovery, локальная устойчивость и изменение шага интегрирования;
- знак мышечного момента, гравитация, закреплённое плечо, reset и seed;
- отдельные ROM/load/stretch penalties и coactivation при неизменной позе;
- признаки очереди задержки и совместимость observation/action contract;
- причинность predictor, разделение эпизодов и train-only normalization;
- сохранение/загрузка predictor и отклонение несовместимых артефактов;
- JSON/SI, отсутствие NaN/Infinity и разделение activation/force;
- перенос PPO + metadata + bundled predictor в новую папку и последующий запуск.

Тест PPO условный: если train extra не установлен, он пропускается. Остальные
проверки не требуют PyTorch. Workflow GitHub Actions устанавливает base/dev и
проверяет CPU-среду; факт локального прохождения не подменяет его отдельный статус.

Первый CI-прогон выявил различие около `6.94e-18` между одиночным и пакетным
матричным вычислением на другой BLAS/CPU сборке. Тест исправлен: JSON roundtrip
по-прежнему проверяет точное совпадение при одинаковой форме входа, а сравнение
GEMV/GEMM использует `rtol=1e-14, atol=1e-15`. Физика и predictor не изменялись.

## Повторная проверка установки после ошибки ensurepip

После сообщения пользователя о системном Python 3.10 проверен новый
`scripts/setup.sh` в отдельной копии репозитория, начиная с неполной `.venv`:

- uv 0.13.4 скачал managed CPython **3.12.15** и создал окружение с pip **26.2.2**;
- editable-установка `.[dev]` завершилась; `pip check` не нашёл конфликтов;
- marker из старой `.venv` сохранился в уникальном backup;
- повторный запуск использовал исправную `.venv`, сохранил marker внутри неё
  и не создал дополнительного backup;
- `OPENBLAS_NUM_THREADS=1 .venv/bin/python3 -m pytest -q`: **75 passed, 1 skipped**,
  4.89 s; PPO-тест пропущен, поскольку train extra не устанавливался;
- 10 s heuristic rollout: RMSE **1.998°**, физических срывов **0**.

Это повторная проверка установки base/dev на Linux с новой patch-версией Python,
а не повторное обучение PPO. Первоначальное окружение 3.12.14 и результаты
обучения ниже сохранены как отдельный проведённый эксперимент. В CI также
добавлено создание неполной `.venv` и проверка её сохранения перед установкой.

## Проверка связи тканей с механикой

Два 10-секундных сценария сравнивались при dt=1 ms и 0.5 ms, с неизменными
20 ms control period, 40 ms delay, seed и начальными условиями.

| Максимальное различие | Heuristic controller | Постоянное excitation |
|---|---:|---:|
| Угол | 0.00137 rad | 0.000914 rad |
| Угловая скорость | 0.00921 rad/s | 0.00579 rad/s |
| Сила одного сухожилия | 0.110 N | 0.0635 N |

Постоянная команда: `[0.15, 0.15, 0.15, 0.04]`. Все четыре прогона дошли до
time limit без физического termination. Проверены знак мощности по virtual work,
локальный residual и неотрицательность упругих энергий/диссипации.

Это проверка двух сценариев; она не доказывает сходимость или устойчивость для
произвольной жёсткости, груза и шага. First-order split оставляет небольшое
различие времени геометрии и последней рассчитанной силы; см. [model.md](model.md).

## Диагностический контроллер

```bash
python3 -m myo rollout --seconds 10 --controller heuristic --seed 0 \
  --output runs/curl-smoke.jsonl
```

Результат этого запуска:

- RMSE угла **1.998°**;
- RMSE угловой скорости **0.08335 rad/s**;
- физических срывов **0**;
- mean reward **0.9830**.

Это результат heuristic с preview известной команды, а не обученной сети.
Он подтверждает, что упрощённой системой можно выполнить исходную задачу.
Первоначальные дополнительные прогоны seed 1/2 дали около 3.10°/2.79° без срыва.

Зафиксированное wall time headless-прогона — около 0.86 s на 10 s симуляции.
Mean `env.step` около 1.54 ms, p95 около 2.23 ms; один вызов включает 20
физических подшагов. Это наблюдение в текущем окружении без целевого аппаратного
профиля. Оно не является обещанием FPS, точным бюджетом одного `mj_step` или
производительностью полного тела/Unreal. CLI сообщает состав измерения.

## Предиктор

```bash
python3 -m myo.predictor collect --episodes 8 --horizon 0.1 \
  --controller heuristic --output runs/predictor-data-smoke.npz
OPENBLAS_NUM_THREADS=1 python3 -m myo.predictor train \
  --data runs/predictor-data-smoke.npz --epochs 100 \
  --output runs/predictor-smoke.json
```

Получено 3968 пар, 2976 train samples и 992 validation samples. Train episodes:
`0,1,3,5,6,7`; validation episodes: `2,4`. Сеть: 185 входов → 32 tanh units →
два выхода. Горизонт 100 ms, или пять шагов контроллера.

| Прогноз | Validation RMSE ошибки угла, rad | Validation RMSE ошибки скорости, rad/s |
|---|---:|---:|
| Ошибка равна нулю | 0.055267 | 0.132163 |
| Текущая ошибка сохранится | 0.015765 | 0.167786 |
| Обученный predictor | **0.008595** | **0.029475** |

Лучший epoch — 100; validation использовалась для выбора весов. Независимого
test set в этом smoke-опыте нет. Эти результаты относятся к heuristic policy,
которая собрала данные. Они **не доказывают**, что добавление predictor улучшает
PPO, компенсирует все задержки или переносится на другую задачу.

Wrapper проверен в реальном цикле env. С данным обученным predictor выполнен
отдельный PPO smoke на 128 переходов, сохранение артефактов и 2-секундный rollout
загруженной политики. Это проверка совместного пути, а не сравнение качества:
длительность, seed и объём обучения отличаются от остальных опытов.

## PPO: короткий smoke и стандартный бюджет

```bash
python3 -m myo.train --steps 4096 --seed 0 --threads 1 \
  --output runs/ppo-smoke.zip
python3 -m myo rollout --controller runs/ppo-smoke.zip --seconds 10 --seed 100
```

После 4096 transitions checkpoint успешно загрузился. На одном 10-секундном
эпизоде seed 100: RMSE **53.07°**, mean reward **0.1557**, физических срывов 0.
**Этот короткий запуск не освоил движение.** Отсутствие аварийного termination
не считается успехом tracking; сравнение с heuristic выше это показывает.

Затем выполнен стандартный бюджет из README:

```bash
python3 -m myo.train --steps 100000 --seed 0 --threads 1 \
  --output runs/ppo-100k.zip
python3 -m myo rollout --controller runs/ppo-100k.zip --seconds 10 --seed 100
```

Фактически выполнено **100096 transitions** (782 полных PPO rollout chunks).
Политика без predictor оценена детерминированно на трёх 10-секундных эпизодах;
для сравнения heuristic проверен на тех же seed и конфигурации.

| Seed | PPO RMSE угла | Heuristic RMSE угла | PPO mean reward | Физические срывы PPO |
|---|---:|---:|---:|---:|
| 100 | 6.97° | 2.79° | 0.8255 | 0 |
| 101 | 7.52° | 3.12° | 0.8048 | 0 |
| 102 | 8.38° | 3.30° | 0.7731 | 0 |

Контроллер научился выполнять tracking в этом ограниченном опыте и улучшился
относительно короткого запуска. Heuristic остаётся точнее. Три эпизода не
устанавливают статистическую надёжность, естественность движения или перенос
на другой груз/жёсткость. Для приёмки D1 нужен более широкий заранее заданный
test set; predictor здесь не использовался, поэтому сравнение его пользы ещё впереди.

## Простое окно демо и схема активаций

После добавления `python3 -m myo demo` выполнен полный локальный набор:
`OPENBLAS_NUM_THREADS=1 .venv/bin/python3 -m pytest -q` — **80 passed**, одно
прежнее предупреждение SB3, 7.64 s. Четыре новые проверки покрывают:

- неизменность физической model/data и мышечных состояний при построении сцены;
- изменение только цвета при изменении активаций в фиксированной позе;
- отсутствие шагов физики и вызовов policy на паузе, полный reset и остановку
  цикла после закрытия;
- закрытие до первого шага с корректным JSON без NaN/Infinity.

Последние две проверки используют настоящий env/controller и подменяют только
окно. Геометрия проверена на настоящих `MjvScene`. Кости и мышцы нарисованы для
нескольких углов, в том числе 45°, 75° и 105°. Через `MUJOCO_GL=egl` отрисован
и визуально просмотрен кадр 960×720 с теми же геометрией, цветовой шкалой и
текстом, что использует новое окно. При отрисовке снимок телеметрии не изменился.
EGL здесь использует программный Mesa; это проверка изображения, не GPU/FPS.

Исходники физики и XML не изменены; `dynamics_sha256` остался
`5e2f6d93f18abb3a4c4bf24b5da110f15d9f1a9c2160ecf6301ea5a58d8e98ea`.
Ранее обученный `ppo-100k.zip` загружен и прошёл короткий 0.1 s rollout без срыва.

**Настоящее GLFW-окно и реальные события мыши/клавиатуры здесь не проверены:**
дисплей отсутствует, а попытка получить Xvfb через зеркало пакетов завершилась
HTTP 503 до загрузки. Headless EGL и проверки цикла не подменяют эту проверку.

Описание нового вида и управления: [demo.md](demo.md).

## Исправление крепления brachialis и проверка активаций

В первой схеме дистальный маркер brachialis находился на расстоянии 16.769 mm
от оси локтевой кости. При радиусах стержня 7.5 mm и маркера 4.5 mm между
поверхностями оставался зазор около **4.77 mm**. Теперь точка крепления
вычисляется на поверхности именованной `ulna`; отрисовка кости и вычисление
крепления используют общее описание её оси.

`OPENBLAS_NUM_THREADS=1 .venv/bin/python3 -m pytest -q`: **83 passed**, прежнее
предупреждение SB3, 7.76 s. Новый параметризованный тест при 45°, 75° и 105°
измеряет расстояние между реальными геометриями маркера и кости в `MjvScene`,
а не повторяет вычисление вспомогательной функции крепления. Кадры этих трёх
поз через EGL просмотрены; часть соединения при 45° перекрывается шарниром
и проксимальной частью кости, но разрыва больше нет.

Заново отрисован `render_frame()` после 100 шагов стандартного heuristic,
`seed=0`, t=2 s: угол 61.196°; снимок телеметрии при отрисовке не изменяется.
Обновлён `docs/images/arm-demo.png`. Физическая модель и контроллер не изменены;
`dynamics_sha256` совпадает с указанным выше.

Отдельный 10 s прогон `PreacherCurlEnv()` с `reset(seed=0)` и
`step(env.baseline_action())` подтвердил причину одинаковых цветов:
возбуждения и активации трёх сгибателей совпадают на всех 500 шагах, а их силы
различаются. Снимок того же прогона при t=2 s:

| Мышца | Активация | Сила сухожилия |
| --- | ---: | ---: |
| `biceps_long` | 12.9187% | 48.8606 N |
| `biceps_short` | 12.9187% | 41.0423 N |
| `brachialis` | 12.9187% | 61.6282 N |
| `triceps` | 1.5000% | 10.4062 N |

На всех 500 шагах этого эпизода возбуждение трицепса оставалось 0.015 —
это заданный фон в heuristic. Его ненулевую активность в данном прогоне
нельзя приписывать обученной стабилизации или отдельному импульсу торможения.
Это наблюдение относится к конкретной конфигурации и seed; в коде есть ветка
увеличения команды разгибателя при отрицательном остаточном требуемом моменте.

## Ревизия тестов — 2026-10-30

По запросу владельца набор сокращён с 83 до **38 тестовых случаев**
(53 → 36 функций). Изменение: `23162e641b737766770d0cda69704770b059c4f5`.
Удалены преимущественно матрицы ошибочных входов и отдельные проверки
копирования; связанные проверки включены в существующие сценарии.
Код `src/`, физическая модель и защитные проверки входов не менялись.
Обоснование отбора и дальнейшие правила — [tests/README.md](../tests/README.md)
и раздел Proportionate testing в [AGENTS.md](../AGENTS.md).

[GitHub Actions, run 41079583728](https://github.com/boris-shurygin/myo/actions/runs/41079583728)
успешно выполнил установку с восстановлением неполной `.venv`, затем:

```bash
.venv/bin/python3 -m pytest -q
.venv/bin/python3 -m myo rollout --seconds 1 --controller heuristic
```

Результат CI: **37 passed, 1 skipped in 4.31 s**. PPO-проверка пропущена,
поскольку CI устанавливает base/dev без train extra. Односекундный headless
rollout завершился без физических срывов; RMSE угла 1.542935°.
Окружение этого запуска: Ubuntu 26.04.1, Python 3.12.16, MuJoCo 3.17.0,
Gymnasium 1.4.0, NumPy 2.5.4. Этот запуск не повторяет обучение PPO и не проверяет
настоящее графическое окно. Локально проверены разбор и компиляция пяти
изменённых тестовых модулей; runtime-проверку выполнил CI, а не локальный sandbox.

Сокращение уменьшает покрытие ошибочных входов; оно не объявляется сохранением
полного прежнего покрытия. Числа 76/80/83 выше относятся к историческим запускам.

## Что ещё не проверено

- Польза predictor для качества/устойчивости обученной policy при сопоставимом бюджете.
- Перенос на MyoSuite/MuscleMimic, GPU backend и анатомические модели.
- Native GLFW viewer и реальные события ввода в текущем окружении без дисплея.
- UE receiver, active Flesh assets, обученный ML Deformer и executable UE build.
- Обмен по UDP с реальным Unreal-получателем; проверены encoder/контракт.
- Full-body движение, естественность и несколько одновременных задач.
- Физиологическая калибровка параметров и любых нагрузочных/болевых proxies.

Демо 1 целиком пока не завершено. Критерии перехода к приседу определены в
[roadmap.md](roadmap.md); текущий отчёт не снимает эти критерии.

