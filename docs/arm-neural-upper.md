# Нейросеть верхнего уровня над замороженным мышечным контроллером

Этап3 [иерархии ускорений](arm-acceleration-hierarchy.md),2026-11-11.
Реализация: `src/myo/arm_neural_upper.py`,runner
`scripts/train_arm_neural_upper.py`. Нижний actor —успешный
`runs/arm_hierarchy/inverse_cold_grid/best.zip`;его weights заморожены.

## Исполнение

```
состояние + эталон q/v с preview
          ↓
нейронный upper → a_cmd на интервал40–60 ms
          ↓ сразу,без дополнительного ожидания
замороженный neural lower → 6 возбуждений
          ↓ транспорт40 ms + активация + упругость
MuJoCo → новое состояние
```

Очередь заданий ускорения не добавляет вторую транспортную задержку.
Нижняя сеть видит новое задание сразу,чтобы выдать мышечную команду заранее.
Эвристическое предложение,которое ещё вычисляет внутренний стенд,маскируется
во входе upper и перезаписывается **до** вычисления любого muscle excitation.
Тест изменяет это предложение независимо от референса и проверяет неизменный
upper-input. В исполнении нейронной иерархии готовый ответ эвристики не используется.

Upper-input151 канал:прежний147-канальный вход lower с due40 proposal=0,
плюс `q_ref(t+50ms)-q`, `v_ref(t+50ms)-v`,среднее `a_ref(40..60ms)/5`,
последнее **измеренное** ускорение/5. Это известное будущее задания;
реальное будущее физики не подаётся ни actor,ни critic.

Actor/critic —раздельные4×128 SiLU. Upper выдаёт одно Gaussian-действие,
которое при исполнении ограничивается[-1,1] и переводится в[-20,20] rad/s².
Lower остаётся детерминированным Beta-mean;параметры его распределения не
дообучаются. Шум PPO теперь относится к **ускорению**,а не шести мышцам.
Начальный log_std−3 соответствует≈1 rad/s² до ограничения;std обучается.

## Обучение и отбор

Первый этап —warmup upper на командах эвристики. Teacher выполняет11 учебных
траекторий через замороженный lower. Добавляется синтетическое варьирование
известных **целей**:σ ошибки q0,1 rad,v0,2 rad/s,a_ff0,5 rad/s²;
физические ощущения при этом не подменяются. Label пересчитывается той же
PD-формулой,что у исходного upper. Это imitation,не самостоятельное RL.
Adam1e−3,batch256,2000 updates;native validation каждые500.
Два последних flush-интервала не являются новыми заданиями и исключены
из warmup-пар. Их ошибочное включение обнаружил runtime timing-check;
первый запуск завершился до warmup и повторён после исправления.

Затем PPO самостоятельно оптимизирует реальное движение:

```
reward = 0.1 * (0.85 * anchored_log_position(q-q_ref)
               + 0.15 * exp(-((v-v_ref)/0.5)^2)
               - physical_failure)
```

Выход эвристики больше не является целью loss. Штрафа за абсолютную скорость
нет;на движении нужна именно скорость эталона. Экономия усилий пока выключена.
RMSprop/LR1e−5,4×256 rollout,batch128,10 epochs,γ0,99,GAE λ0,95,
clip0,2,target KL0,03,entropy0,vf0,5,max_grad_norm0,5. Бюджет32768 transitions.
Отбор —меньше native failures,больше pass,затем меньше validation RMSE угла.
Один общий best охватывает imitation и PPO;плохой PPO не вытесняет warmup.

Train11,validation5 и известный test3 —те же полные траектории,что в последнем
опыте lower. Mass1 kg,наклон120°,capacity1,6 s,исходная механика без упрощений.
Test уже известен;успех не равен новой независимой проверке обобщения.

## Контракт и проверки

Upper checkpoint self-contained:содержит **обе** neural policies и полную
identity lower/физики/входа. Исходный lower-файл после загрузки не требуется.
Сохраняется provenance исходного lower SHA. Strict loader проверяет bytes,
геометрию,нормировку,obs-order,action mapping,источники и reference identity.

Behavioral tests:отсутствие heuristic answer во входе,точный native replay
посредством lower,досчёт двух последних команд,неизменные lower weights после
настоящего PPO-update upper,self-contained reload и отклонение изменённых
байтов checkpoint. Моторная задержка/активация/series elasticity не меняются.

Запуск с прежними ограничениями CPU12–15,nice19,ionice idle,Torch/BLAS1:

```bash
.venv/bin/python3 scripts/train_arm_neural_upper.py \
  --lower runs/arm_hierarchy/inverse_cold_grid/best.zip \
  --out runs/arm_hierarchy/neural_upper --warm-updates 2000 --ppo-steps 32768
```

## Кривая верхней сети

| Фаза | Updates / transitions | Validation pass | Mean RMSE угла, ° |
| --- | ---: | ---: | ---: |
| Imitation | 500 updates | 5/5 | 1,279 |
| Imitation | 1000 updates | 5/5 | 1,030 |
| Imitation | 1500 updates | 5/5 | 1,021 |
| Imitation | 2000 updates | 5/5 | 1,059 |
| PPO | 16384 transitions | 0/5 | 3,554 |
| PPO | 32768 transitions | 3/5 | 3,067 |

PPO не вытеснил успешный warmup. EV critic на
первом блоке уже высока,но это не проверка правильного ранжирования действий
через advantage и не гарантия полезного policy-update. Нельзя приписывать
все проблемы одному лишь неточному V. Последний PPO-блок лучше первого,
но хуже warmup:это остановка по бюджету,не установленное плато.

## Итог

Best —imitation update1500,lower по-прежнему frozen inverse update2000.
В исполнении **обе сети нейронные**,эвристика не подаёт готовый ответ.
В выбранном checkpoint critic upper ещё начальный и в исполнении не
используется. Высокая EV относится к отдельной,last-PPO ветви;перенос её
critic в лучший imitation-actor не выполнялся. Это важно для следующего
продолжения обучения:нельзя считать value-function best уже обученной.

| Набор | Pass | Mean RMSE угла, ° | Mean RMSE ускорения lower, rad/s² |
| --- | ---: | ---: | ---: |
| Train | 11/11 | 1,030 | 0,494 |
| Validation | 5/5 | 1,021 | 0,545 |
| Известный test | 3/3 | 1,043 | 0,528 |

Physical failures0. Native gate по-прежнему ограничивает position RMSE≤3°,
velocity error≤10°/s,acceleration error≤3 rad/s² после0,5 s. Это локальная
демонстрация120–135°,1 kg/наклон120°/6 s,не самостоятельное освоение движения
через RL и не общий контроллер всего тела. Экономия усилий не обучалась.

```bash
.venv/bin/python3 scripts/demo_arm_hierarchy.py --motion oscillation --view
.venv/bin/python3 scripts/demo_arm_hierarchy.py --motion test_lower --view
```

Default —`runs/arm_hierarchy/neural_upper/best.zip` и metadata. Чтобы сравнить
с эвристическим upper при том же нижнем actor,задать
`--controller runs/arm_hierarchy/inverse_cold_grid/best.zip`. Обаcheckpoint
локальные ignored. Новое окно в этом сеансе не открывалось.
