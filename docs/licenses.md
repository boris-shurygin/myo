# Лицензии, происхождение и состав демо

Дата проверки первичных источников: **2026-10-29**.

## Статус нашего проекта

Владелец пока не выбрал лицензию кода `boris-shurygin/myo`.
Этот документ не назначает новую лицензию и не заменяет файл `LICENSE`.
Лицензия зависимости не становится автоматически лицензией нашего кода.

V0 использует собственную условную геометрию руки, четыре мышечные тяги
и аналитически заданную траекторию. Это не скопированные определения
MoBL-ARMS, MyoArm или MuscleMimic. AMASS, SMPL-family assets, чужие motion
clips и готовые checkpoint в распространяемый V0 не входят.

Такой состав отделяет проверку механики от доступа к сторонним данным.
Это не заявление о выборе лицензии для исходного кода или результатов V0.

## Код библиотек и будущих интеграций

| Компонент | Первичный файл лицензии | Значение для проекта |
| --- | --- | --- |
| MuJoCo | [Apache-2.0](https://github.com/google-deepmind/mujoco/blob/main/LICENSE) | Разрешительная лицензия кода физического движка |
| Gymnasium | [MIT](https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE) | Интерфейс среды и его реализация |
| NumPy | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | Численные массивы и MLP V0; у бинарных сборок могут быть дополнительные notices |
| Stable-Baselines3, если установлен для PPO | [MIT](https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE) | Опциональная библиотека обучения; не лицензия policy/data |
| PyTorch, если установлен для PPO | [BSD-style, полный LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) | Вычисления нейросети; бинарные сборки и зависимости имеют собственные notices |
| MyoSuite | [Apache-2.0](https://github.com/MyoHub/myosuite/blob/main/LICENSE) | Код будущих сред; состав ассетов проверяется отдельно |
| myo_sim | [Apache-2.0 верхнего уровня](https://github.com/MyoHub/myo_sim/blob/main/LICENSE) | Не отменяет условий upstream моделей |
| MuscleMimic | [Apache-2.0](https://github.com/amathislab/musclemimic/blob/main/LICENSE) | Код framework; README отдельно выделяет данные, модели и weights |
| URLab | [Apache-2.0](https://github.com/URLab-Sim/UnrealRoboticsLab/blob/main/LICENSE) | Возможный будущий плагин; не лицензия Unreal Engine |

При включении стороннего кода/бинарных зависимостей в релиз сохраняются
относящиеся к ним copyright notices, тексты лицензий и применимые NOTICE.
Для Apache-2.0 изменения заимствованных файлов отмечаются; конкретный
состав notices сверяется с закреплённой версией dependency.
Этот список — реестр выбранных технологий, а не полный SBOM пакета.

## Модели: Apache badge недостаточен

В [NOTICE musclemimic_models](https://github.com/amathislab/musclemimic_models/blob/main/NOTICE)
перечислены источники частей тела. Для MoBL-ARMS, используемой в MyoArm /
MyoHand, указана **BSD 3-Clause с ограничением non-commercial**. Включены
research, academic, evaluation и personal uses. Верхнеуровневая Apache-2.0
не даёт основания считать всю цепочку модели коммерчески свободной.

Это ограничение определения модели нельзя автоматически переносить на
каждый mesh: тот же NOTICE отдельно указывает CC BY 3.0 для стандартной
костной геометрии OpenSim и MIT для ряда других исходных моделей.
Практическая единица аудита — конкретный файл и его происхождение.

Для личного некоммерческого эксперимента модель может быть допустима по
своим условиям. Для будущей продажи потребуется определить фактически
использованные части и получить подходящие права либо заменить их.
Собственный motion capture не меняет лицензию анатомической модели.

## Данные и готовые политики

| Источник | Проверенные условия | Решение V0 |
| --- | --- | --- |
| [AMASS](https://amass.is.tue.mpg.de/license.html) | Некоммерческие research/education/art; запрет передачи данных третьим лицам; запрет обучения для коммерческого применения | Не скачивать автоматически и не включать в релиз |
| [MuscleMimic retargeted](https://huggingface.co/datasets/amathislab/musclemimic-retargeted) | Условия AMASS распространяются на данные и derivative files; коммерческое использование обученных на них models/methods запрещено | Использовать аналитический reference |
| [MuscleMimic demo_dataset](https://huggingface.co/datasets/amathislab/demo_dataset) | Текст карточки указывает AMASS/non-commercial, несмотря на Apache badge | Не считать маленький demo archive свободным для перепаковки |
| [mm-fullbody-base](https://huggingface.co/amathislab/mm-fullbody-base) | На проверенной карточке нет самостоятельного разрешения на распространение/commercial use weights | Не включать готовые веса |

Бесплатность скачиваемого демо не снимает запрет перераспространения
AMASS-производных файлов. Нельзя объявить checkpoint разрешённым только
потому, что код обучения имеет Apache-2.0. Условия конкретных весов и
данных обучения проверяются отдельно; отсутствие явного текста не трактуется
как разрешение.

Если позднее понадобятся SMPL/SMPL-X fitting и ассеты, их проверка отдельная:
[SMPL-X model license](https://smpl-x.is.tue.mpg.de/modellicense.html).
В V0 fitting и эти файлы не нужны. Аналогично, UE-проект будет подчиняться
[Epic EULA](https://www.unrealengine.com/en-US/eula/unreal) и условиям каждого
включённого asset; лицензия URLab их не заменяет.

## Что фиксировать при добавлении каждого внешнего ассета

1. Исходный URL, автор/правообладатель, commit или версия и checksum файла.
2. Сам текст лицензии/условий и обязательная атрибуция.
3. Разрешения для локального эксперимента, обучения и распространения.
4. Разрешения для коммерческого применения данных и производных weights.
5. Изменения: конверсия, ретаргетинг, зеркалирование, вырезанная часть модели.

Модели, motion data, checkpoints и визуальные meshes учитываются раздельно.
Если конкретное разрешение не установлено, asset остаётся вне релиза до
выяснения. Публикация собственного исходного V0 не требует добавлять к нему
сторонние данные или принимать за пользователя gated download agreements.
