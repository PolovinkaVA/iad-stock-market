# Установка и воспроизведение MSFT

Проверено 16 октября 2026 года: новое окружение Python 3.12.8 на том же Windows-компьютере, отдельная копия кода, сохранённый исследовательский снимок. Готовые модели и SQLite не переносились. Восстановлены шесть моделей и объяснения 252 дат; максимальная разность тестовых прогнозов — 0 USD. В основной и отдельной копии прошли все 33 теста, включая Streamlit AppTest. Это не проверка другой ОС или повторной загрузки неизменных данных из сети.

## Что требуется для точного восстановления

Код, конфигурации, протокол и сводки находятся в Git. Полные тексты, база, снимок и модели хранятся локально. Для опубликованного опыта нужен файл `data/processed/msft/snapshot.json` из сохранённых материалов проекта. Его содержательный хеш в манифесте — `686028db65227ac1bec9422844df566855a25d622528f67d328ae9fe90953915`. Он вычисляется функцией `sha` над JSON-объектом, а не над байтами форматированного файла.

Без этого снимка репозитория достаточно для изучения кода и сводок, но недостаточно для точного восстановления опубликованных чисел. Текущая загрузка может вернуть исправленные котировки и тексты.

Из корня проекта после переноса снимка:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/prepare_msft.py --snapshot data/processed/msft/snapshot.json
.\.venv\Scripts\python.exe scripts/train_msft.py --stage test --rebuild-artifacts
.\.venv\Scripts\python.exe scripts/explain_msft.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run scripts/dashboard_msft.py --server.address 127.0.0.1 --server.port 8501
```

Параметры берутся из опубликованного протокола. `--rebuild-artifacts` заново создаёт модели и предсказания. При существующем test-отчёте все метрики сверяются с ним с абсолютным допуском 10⁻¹⁰: при несовпадении прежние модели и предсказания не заменяются. Сам опубликованный test-отчёт сохраняется без изменения. Обычный `--stage test` без этого ключа читает сохранённый результат и не восстанавливает отсутствующие модели.

Исторический интерфейс работает по снимку, предсказаниям и объяснениям без базы SQLite. Для рабочего обновления сначала заполняется база командой `scripts/ingest_msft.py`, затем выполняется `scripts/update_msft.py --refresh`. Общая команда `run_msft.ps1` включает рабочий расчёт, поэтому для восстановления только исторического интерфейса выше приведён прямой запуск Streamlit.

## Независимая копия для проверки

Если в исходном проекте уже имеются сохранённый снимок и тестовые предсказания, команда создаёт отдельный каталог без переноса готовых моделей:

```powershell
.\.venv\Scripts\python.exe scripts/verify_msft_reproduction.py --output tmp/reproduction_check
```

Выходной каталог должен быть новым подкаталогом `tmp/`; существующий не очищается. Скрипт заново строит признаки, повторяет validation по исходным правилам, восстанавливает модели, сравнивает предсказания с исходными и строит объяснения. Результат записывается в `reports/msft_reproduction.json`. Для полной проверки нового окружения сначала создайте отдельный venv, установите туда lock-файл и запустите скрипт его Python.

## Эксперимент со свежей загрузкой

Новые версии источников нельзя молча записывать поверх опубликованного эксперимента. После загрузки создайте отдельную конфигурацию:

```powershell
.\.venv\Scripts\python.exe scripts/ingest_msft.py --end 2026-10-15
$experiment = Get-Content config/msft_experiment.json -Raw | ConvertFrom-Json
$experiment.processed_dir = 'data/processed/msft_new'
$experiment.model_dir = 'models/msft_new'
$experiment.reports_dir = 'reports/generated/msft_new'
$experiment | ConvertTo-Json -Depth 10 | Set-Content config/msft_new.json -Encoding utf8
.\.venv\Scripts\python.exe scripts/prepare_msft.py --config config/msft_new.json
.\.venv\Scripts\python.exe scripts/train_msft.py --config config/msft_new.json --stage validate
.\.venv\Scripts\python.exe scripts/train_msft.py --config config/msft_new.json --stage test
.\.venv\Scripts\python.exe scripts/explain_msft.py --config config/msft_new.json
```

Сохраните причину повторного опыта и новый хеш данных. Уже просмотренный период остаётся ретроспективной проверкой. Штатный дашборд читает `config/msft_experiment.json`; приведённые отдельные каталоги автоматически в него не переключаются.

## Сборка отчёта

Готовый [DOCX](../reports/Отчет_ИАД_MSFT_Половинка_ВА.docx) доступен в репозитории. Текст — `docs/report_msft.md`; числовые таблицы и графики формируются из сохранённых результатов скриптом `scripts/build_msft_report_assets.py` (pandas, numpy, reportlab, шрифт Arial в Windows). `scripts/render_report_figures.cjs` переводит SVG в PNG через пакет Node `sharp`. Затем `scripts/build_msft_report.py --reference <образец.docx>` собирает документ через python-docx, используя только титульный лист образца. Сам образец и его основной текст в репозиторий не включаются.

`scripts/export_report_word.ps1` обновляет содержание и поля через установленный Microsoft Word и экспортирует PDF для проверки. Вёрстка проверена по изображениям всех страниц; PDF и изображения проверки остаются в `tmp/`. Зависимости оформления используются отдельно от окружения моделей и не нужны для запуска дашборда.
