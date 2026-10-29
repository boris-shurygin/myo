# Установка на Linux

## Версия Python

Проверенная среда проекта — Python 3.12. Файл `.python-version` фиксирует эту
ветку для инструментов управления окружением. Минимальная версия в метаданных
пакета — 3.11; Python 3.10 не поддерживается текущим контрактом проекта.
Полный `requirements-verified.txt` воспроизводит окружение Python 3.12.

Команда `python3` вне виртуального окружения запускает системный Python, версия
которого зависит от ОС. Поэтому создавать окружение через произвольный
`python3 -m venv` недостаточно.

## Рекомендуемый способ: uv и отдельный Python 3.12

Установить uv официальным standalone installer, без sudo:

```bash
(
  set -euo pipefail
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
)
```

Из каталога `myo`:

```bash
(
  set -euo pipefail
  export PATH="$HOME/.local/bin:$PATH"
  bash scripts/setup.sh
  .venv/bin/python3 -m pytest -q
  .venv/bin/python3 -m myo rollout --seconds 10 --controller heuristic
)
```

`setup.sh` использует `uv venv --managed-python --python 3.12 --seed`.
При необходимости uv скачивает свой Python в пользовательский каталог;
системный `/usr/bin/python3` и apt-пакеты не меняются. Системный ensurepip
этому способу не требуется. Зависимости ставятся через `uv pip` с явным
`--python .venv/bin/python3`; pip также устанавливается внутрь окружения для
дальнейших команд `python3 -m pip` из README.

Если `.venv` отсутствует, она создаётся. Если она неполная или использует другую
ветку Python, скрипт перемещает её в уникальный `.venv-backup.*/.venv` и создаёт
новую. Исходники, данные и веса остаются на месте. Исправная `.venv` Python 3.12
используется повторно. При ошибке скачивания или установки скрипт завершается
с ненулевым кодом; повторный запуск продолжает подготовку окружения.

Для дальнейшей работы в текущем терминале:

```bash
source .venv/bin/activate
python3 --version
python3 -m pip --version
python3 -m myo rollout --seconds 20 --viewer --realtime
```

Первые две команды проверки должны показать Python 3.12.x и pip внутри
`myo/.venv`. Viewer требует графического дисплея; тесты и headless rollout — нет.

## Если Python 3.12 уже установлен без uv

У интерпретатора должен работать модуль venv/ensurepip. Названия и доступность
apt-пакетов зависят от версии дистрибутива; не следует устанавливать
`python3.10-venv` в расчёте получить Python 3.12.

Следующие команды предназначены для каталога без существующей `.venv`:

```bash
(
  set -euo pipefail
  test ! -e .venv
  test ! -L .venv
  python3.12 -m venv .venv
  .venv/bin/python3 -m pip install --upgrade pip 'setuptools>=80'
  .venv/bin/python3 -m pip install -e '.[dev]'
)
```

## Разбор ошибки первого запуска

- `ensurepip is not available`: создание окружения остановилось до установки
  проекта. Недособранную `.venv` нельзя считать активированной.
- `activate: No such file or directory`: активация не состоялась. Последующие
  отдельные строки shell продолжают выполняться вне окружения.
- `Defaulting to user installation`: pip начал установку в пользовательский
  каталог системного Python, а не в `.venv`.
- `missing build_editable`: editable-установка также не состоялась. В текущем
  проекте указан `setuptools.build_meta` с `setuptools>=80`, который поддерживает
  этот hook. Без полного лога сборки нельзя установить, почему был использован
  другой/несовместимый backend. Подготовка новой среды через uv обходит старый
  системный pip; менять проект на legacy `setup.py` для этого не требуется.
- `No module named gymnasium` и `No module named myo`: ожидаемые последствия
  того, что установка проекта и зависимостей не завершилась.

Если установка снова завершится ошибкой, для разбора нужен вывод начиная с
**первой** ошибки. Команды тестов и запуска выполняются после успешного setup.

## Первичные источники

- [Официальная установка uv](https://docs.astral.sh/uv/getting-started/installation/).
- [Управление версиями Python через uv](https://docs.astral.sh/uv/guides/install-python/).
- [Создание окружений и явный выбор интерпретатора](https://docs.astral.sh/uv/pip/environments/).
- [Editable installs в setuptools](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).
