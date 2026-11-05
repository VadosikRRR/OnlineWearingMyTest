# OnlineWearingMyTest — базовый KGI от исходных фотографий

Отдельная Python-библиотека и [Kaggle-ноутбук](notebooks/kaggle_kgi_end_to_end.ipynb).
Вход: изображение одежды, изображение человека, строка типа одежды.
Выход: RGB-примерка 1024×768, промежуточные карты, маски и точки.

```text
Одежда → GarmentIQ HRNet ───────────────────┐
Человек → OpenPose → 10 суставов ──────────┼→ KG → 32 целевые точки
Человек → SCHP LIP-20 → KGI-13 → agnostic ┘          ↓
                                   PG → целевой parse
                                   TPS → композиция + keep mask
                                   SCI / RePaint → RGB
```

KG, PG и SCI используют архитектуры и веса исходного KGI. Нужный inference-код
включён в пакет: клонировать KGI, SCHP, OpenPose или GarmentIQ отдельно не нужно.
GarmentIQ и OpenPose используются через PyPI, SCHP — через ONNX Runtime.
Ни одна модель не скачивается конструктором или вызовом системы.

**Статус:** реализована цепочка и проверки без весов. Локально успешно выполнены
компиляция, проверка синтаксиса Python 3.11 и unit-тесты. Реальный inference,
установка ML-зависимостей и качество изображений здесь не проверялись.
Подробности: [validation.md](docs/validation.md).

## Запуск в Kaggle

1. Опубликуйте эти файлы в своём `VadosikRRR/OnlineWearingMyTest` либо прикрепите
   архив к Kaggle и укажите `LOCAL_CODE` в ноутбуке. Изменения локальные, push не выполнялся.
2. Прикрепите три файла KGI (KG/PG/SCI) как Kaggle Dataset: [файлы и ссылки](docs/weights.md).
   HRNet/OpenPose/SCHP ноутбук скачает с Hugging Face; для локального режима всех шести
   файлов есть `DOWNLOAD_AUX_FROM_HF=False`.
3. Импортируйте [ноутбук](notebooks/kaggle_kgi_end_to_end.ipynb), оставьте Accelerator **None**,
   включите Internet для установки/streaming, задайте `WEIGHTS_ROOT`.
4. Выполните установку, CPU-проверки и просмотр препроцессинга.
5. Включите `RUN_SINGLE` для одной примерки. Для GPU выберите ускоритель сами,
   задайте `DEVICE="cuda:0"` при создании класса; можно выбрать `SCI_PRECISION="float16"`.
6. Включите `RUN_BATCH` для 1000 записей. Журнал позволяет продолжить прогон;
   для новой сессии сохраните output и задайте `PREVIOUS_RUN`.

Целевая CPU-сессия Kaggle использует **Python 3.13.18** (по проверке пользователя).
Минимум пакета остаётся Python **3.11+**; понижать Python не требуется.
Ноутбук использует текущий kernel. Фактические версии сохраняются в `runtime.json`
и `environment.txt`. Pillow 13 разрешён; после pip проверяется, не остались ли
в памяти модули другой версии. [Исправление UnidentifiedImageError](docs/troubleshooting.md).
Оригинальные настройки SCI — 50 шагов с 10 RePaint-повторами: это существенно больше
50 вызовов сети. Прогон 1k может потребовать нескольких сессий; время/VRAM пока не измерены.
Для первого smoke run предусмотрены 10 шагов и 1 повтор, с другой папкой эксперимента.

## Python API

```python
from pathlib import Path
from online_wearing import Weights, OnlineWearingPipeline

root = Path("/kaggle/input/onlinewearing-weights")
weights = Weights(
    kg=root / "kg/step_299999.pt",
    pg=root / "pg/step_9999.pt",
    sci=root / "sci/ema_0.9999_300000.pt",
    garment_hrnet=root / "hrnet.pth",
    openpose_body=root / "body_pose_model.pth",
    human_parser_lip=root / "parsing_lip.onnx",
)
system = OnlineWearingPipeline(weights, device="cpu")
result = system("cloth.png", "person.jpg", "long sleeve top", seed=42)
result.image                 # PIL.Image RGB
result.save("outputs/example", debug=True)
```

Допустимы пути, PIL-изображения или `uint8` RGB/RGBA-массивы. EXIF-поворот учитывается,
прозрачный фон накладывается на белый. Изображения однократно растягиваются до 768×1024
до всех детекторов; для своих фото предпочтителен заранее подготовленный кадр 3:4.
В кадре должен быть один человек с видимыми плечами/бёдрами; одежда — отдельный товарный снимок.

Модели загружаются лениво, `cpu_offload=True` освобождает устройство между стадиями.
SCHP и TPS исполняются на CPU, устройство torch-моделей задаёт `device`.
Отдельная проверка входов до загрузки SCI:

```python
prepared = system.preprocess("cloth.png", "person.jpg", "vest")
# prepared.parse_lip, prepared.parse_agnostic, prepared.cloth_points, prepared.pose
result = system.infer_prepared(prepared, seed=42)
```

## FITVTO-1k и мерки

Ноутбук потоково читает **первые 1000 записей `eval`** из
[Yuanhao-Harry-Wang/fitvto-100k](https://huggingface.co/datasets/Yuanhao-Harry-Wang/fitvto-100k),
фиксирует commit датасета и сохраняет только эту выборку. Полный 212-GB датасет
не загружается; parquet streaming может прочесть больше 1000 строк на уровне сетевых блоков.
Локальный экспорт: три PNG на запись, относительные пути, `manifest.jsonl` и `source.json`.

Поля изображений — `cloth`, `person`, `target`. `target` поступает только в метрики,
никогда в генератор. MAE/PSNR/SSIM — paired-метрики всего кадра по успешным примерам;
в отчёте отдельно видны ошибки. Это не оценка точности физических размеров.

По согласованному scope мерки **сохраняются для следующего этапа**, но базовый KGI их
не использует: `body_bust`, `body_height`, `body_hips`, `body_waist`, `garment_bust`,
`garment_length`, `garment_sleeve_length` лежат в `measurements_cm`.
Оценивать сантиметры по некалиброванной фотографии система не пытается.

В FITVTO нет колонки категории. Поэтому `DEFAULT_GARMENT_TYPE` — явное допущение,
а `GARMENT_TYPES` позволяет назначить тип каждому ID. Для корректного сравнения
категории следует проверить; автоматической классификации в этой версии нет.

## Совместимость и границы baseline

- `long sleeve top`: 33 точки GarmentIQ → 32 узла по индексам KGI.
- `vest`: 15 точек → повторы в точности по загрузчику KGI.
- `short sleeve top`: 25 точек → экспериментальное схлопывание дистальных узлов рукавов.
  Этот mapping отсутствует в исходном KGI и требует проверки качества.
- Платья, юбки и брюки отклоняются: исходные веса KGI относятся к верху.
- OpenPose из controlnet-aux — реализация семейства PAF/COCO-18, а не точные веса
  исходного Caffe BODY_25. Нужные 10 суставов сопоставлены явно, включая разные индексы бёдер.
- Подготовка agnostic parse/масок восстановлена; совпадение с аннотациями авторов
  не гарантируется. LIP-20 → 13 использует опубликованную таблицу HR-VITON.
- TPS реализован через обратное преобразование и `numpy.linalg.solve`; белая ткань
  сохраняется внутри геометрического контура. Оригинальное отсечение почти белых
  пикселей не применяется. Для сложного фона потребуется отдельная маска товара.

Контракты и источники решений: [architecture.md](docs/architecture.md),
[research.md](docs/research.md), [THIRD_PARTY.md](THIRD_PARTY.md).

## Проверки и CLI

Проверка **без установки библиотек и без весов**, из любого каталога:

```bash
python3 scripts/verify_static.py
```

После установки в Kaggle:

```bash
python -m pip install -e .
python scripts/check_runtime.py
online-wearing prepare data/fitvto_eval_1k --garment-type "long sleeve top"
online-wearing infer --weights weights.json --cloth cloth.png --person person.jpg \
  --garment-type "long sleeve top" --output outputs/one --device cpu
online-wearing evaluate --weights weights.json --manifest data/fitvto_eval_1k/manifest.jsonl \
  --output outputs/eval --limit 1000 --device cpu
```

`weights.json` содержит шесть ключей `Weights`, как в Python-примере.
Ноутбук также создаёт этот JSON. Его исходный генератор — `scripts/build_notebook.py`.
`UPDATE_CODE=True` обновляет существующий чистый checkout до выбранного `REPO_REF`;
при локальных изменениях он останавливается, сохраняя их. Для закреплённой офлайн-копии
можно установить `UPDATE_CODE=False`.
Веса, датасеты, результаты и окружение в git не включаются.
