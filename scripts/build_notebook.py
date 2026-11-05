"""Generate the clean Kaggle notebook with standard Python code cells (no magics)."""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append(dict(cell_type="markdown", metadata={}, source=dedent(text).strip().splitlines(keepends=True)))


def code(text):
    cells.append(dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
                      source=dedent(text).strip().splitlines(keepends=True)))


markdown("""
# OnlineWearing: KGI от двух фотографий до результата

**Kaggle, Python 3.11+. Accelerator: None. Internet: On** для клонирования, установки
и загрузки только 1000 записей FITVTO. Этот notebook сам GPU не включает.
Устройство задаётся только `DEVICE` при создании `OnlineWearingPipeline`.

Цепочка: GarmentIQ HRNet → OpenPose → SCHP LIP-20 → KGI KG → PG → TPS → SCI.
Внешние репозитории моделей не клонируются. Шесть файлов весов загрузите как Kaggle Dataset.
Список файлов и ссылки: `docs/weights.md` в репозитории.

Это восстановленный **базовый** KGI: маски/сопоставления требуют визуальной проверки.
Физические мерки сохраняются, но на генерацию не влияют. Поддержан верх:
`long sleeve top`, `vest`; `short sleeve top` — экспериментальное расширение.
Для платьев/брюк исходные веса KGI не подходят.

Сначала выполните CPU-проверки и просмотр препроцессинга. Полная SCI на CPU медленная;
`RUN_SINGLE` и `RUN_BATCH` включаются явно после проверки. Полный прогон 1000 изображений
может занять несколько Kaggle-сессий. Результаты возобновляются, если сохранить и перенести папку runs.
""")
code('''
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

assert sys.version_info >= (3, 11), "Select a Kaggle Python 3.11+ runtime"
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
REPO = WORK / "OnlineWearingMyTest"
REPO_URL = "https://github.com/VadosikRRR/OnlineWearingMyTest.git"
REPO_REF = "main"  # Better: use a published commit SHA for a reproducible run.
LOCAL_CODE = None  # Alternative: Path('/kaggle/input/onlinewearing-code/OnlineWearingMyTest')
WEIGHTS_ROOT = Path("/kaggle/input/onlinewearing-weights")  # Edit your Kaggle input mount.
EXISTING_SUBSET = None  # Alternative: Path('/kaggle/input/my-fitvto-1k/fitvto_eval_1k')
PREVIOUS_RUN = None  # Optional: saved run directory from an earlier session.

DEVICE = "cpu"  # Explicit opt-in only: change to 'cuda:0' AFTER enabling a Kaggle accelerator yourself.
SCI_PRECISION = "float32"  # Optional 'float16' on an explicitly chosen CUDA device.
SUBSET_SIZE = 1000
DATA_REVISION = "main"  # The resolved HF commit is persisted in source.json.
DEFAULT_GARMENT_TYPE = "long sleeve top"  # Explicit assumption, NOT an automatic classifier.
GARMENT_TYPES = {}  # Per-ID overrides, e.g. {'000000': 'short sleeve top', '000001': 'vest'}.
PREVIEW_INDEX = 0
SAMPLING_STEPS = 50  # Original baseline; for a smoke run set 10 and REPAINT_REPEATS=1.
REPAINT_REPEATS = 10
SEED = 42
RUN_SINGLE = False
RUN_BATCH = False
SUBSET_DIR = WORK / "fitvto_eval_1k"
RUN_DIR = WORK / "runs" / f"kgi_s{SAMPLING_STEPS}_r{REPAINT_REPEATS}"
os.environ.setdefault("HF_HOME", str(WORK / ".cache" / "huggingface"))
print("Python:", sys.version.split()[0], "requested device:", DEVICE)
''')
markdown("""
## 1. Получить код

Сначала опубликуйте подготовленные файлы в своём `OnlineWearingMyTest`, чтобы clone содержал
эту реализацию. Для приватного/ещё не опубликованного репозитория прикрепите архив к Kaggle
и задайте `LOCAL_CODE`: дополнительный репозиторий не нужен.
""")
code('''
if not (REPO / "pyproject.toml").is_file():
    if LOCAL_CODE is not None:
        shutil.copytree(LOCAL_CODE, REPO, dirs_exist_ok=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)
        if REPO_REF != "main":
            subprocess.run(["git", "-C", str(REPO), "fetch", "--depth", "1", "origin", REPO_REF], check=True)
            subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", "FETCH_HEAD"], check=True)
assert (REPO / "src/online_wearing/pipeline.py").is_file(), "Clone/upload the completed implementation first"
if (REPO / ".git").exists():
    print("Code commit:", subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())
''')
markdown("""
## 2. Установка в Kaggle и проверки без весов

Установка выполняется в Kaggle. Локально скачивать ML-библиотеки не требуется.
GarmentIQ закреплён на `0.0.4.11`, controlnet-aux — на `0.0.10`. pip согласует зависимости
PyTorch/torchvision; фактические версии сохраняются. Если Kaggle сообщает о необходимости
перезапуска kernel после обновления бинарных библиотек, перезапустите его и повторите ячейки.
""")
code('''
subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(REPO)], check=True)
sys.path.insert(0, str(REPO / "src"))  # Editable-install path is usable in this same kernel.
subprocess.run([sys.executable, str(REPO / "scripts/verify_static.py")], check=True)
subprocess.run([sys.executable, str(REPO / "scripts/check_runtime.py")], check=True)
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
(WORK / "environment.txt").write_text(freeze)
''')
markdown("""
## 3. Все веса — локальные пути

Положите файлы в `WEIGHTS_ROOT`, сохранив приведённую структуру, или измените каждый путь.
Конструктор проверяет все шесть файлов. Здесь нет скрытого скачивания моделей.
`scripts/download_aux_weights.py` — отдельный необязательный помощник для получения трёх
вспомогательных весов с HF перед прикреплением к Kaggle. Для KG/PG/SCI нужны оригинальные веса KGI.
""")
code('''
from online_wearing import Weights, OnlineWearingPipeline

weights = Weights(
    kg=WEIGHTS_ROOT / "kg/step_299999.pt",
    pg=WEIGHTS_ROOT / "pg/step_9999.pt",
    sci=WEIGHTS_ROOT / "sci/ema_0.9999_300000.pt",
    garment_hrnet=WEIGHTS_ROOT / "hrnet.pth",
    openpose_body=WEIGHTS_ROOT / "body_pose_model.pth",
    human_parser_lip=WEIGHTS_ROOT / "parsing_lip.onnx",
)
weights.validate()
for name, path in weights.as_dict().items():
    print(name, Path(path).stat().st_size // (1024 * 1024), "MiB", path)
(WORK / "weights.json").write_text(json.dumps(weights.as_dict(), indent=2))
pipeline = OnlineWearingPipeline(
    weights, device=DEVICE, sci_precision=SCI_PRECISION, cpu_offload=True,
    sampling_steps=SAMPLING_STEPS, repaint_repeats=REPAINT_REPEATS,
)
''')
markdown("""
## 4. FITVTO: ровно 1000 записей eval, потоковое чтение

В опубликованной схеме изображения называются **cloth, person, target**. Пример в карточке
датасета содержит несогласованное имя `image`; здесь схема проверяется явно.
`person` — входное фото; `target` — эталон исключительно для метрик.
Семь полей мерок в сантиметрах сохраняются в `manifest.jsonl`.

Категории одежды в схеме нет. Перед большим прогоном проверьте товарные фото и задайте
`GARMENT_TYPES` для отличающихся примеров; `DEFAULT_GARMENT_TYPE` записывается как явное допущение.
Изменение категорий/ревизии требует новой папки subset. Выборка — первые 1000 строк без shuffle,
поэтому она воспроизводима, но не является случайной репрезентативной выборкой.
Streaming не скачивает весь датасет (~212 GB); сеть может читать целые parquet row groups.
""")
code('''
from online_wearing.data import prepare_fitvto_subset, read_manifest

if EXISTING_SUBSET is not None:
    manifest = EXISTING_SUBSET / "manifest.jsonl"
else:
    manifest = prepare_fitvto_subset(
        SUBSET_DIR, limit=SUBSET_SIZE, split="eval", revision=DATA_REVISION,
        default_garment_type=DEFAULT_GARMENT_TYPE, garment_types=GARMENT_TYPES,
    )
records = read_manifest(manifest)
assert len(records) == SUBSET_SIZE == 1000
print(json.loads((manifest.parent / "source.json").read_text()))
print("Saved records:", len(records), "measurements:", records[PREVIEW_INDEX]["measurements_cm"])
''')
code('''
from PIL import Image
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(14, 8))
for ax, record in zip(axes.flat, records[:10]):
    with Image.open(manifest.parent / record["cloth"]) as image:
        ax.imshow(image)
    ax.set_title(f"{record['id']}: {record['garment_type']}", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()
record = records[PREVIEW_INDEX]
cloth_path = manifest.parent / record["cloth"]
person_path = manifest.parent / record["person"]
garment_type = record["garment_type"]
print("Preview:", record["id"], garment_type)
''')
markdown("""
## 5. Посмотреть препроцессинг до SCI

Входные кадры приводятся к 768×1024 один раз перед детекторами. На произвольных фото
это resize без сохранения соотношения сторон; для хорошего результата подготовьте кадр 3:4.
OpenPose должен видеть одного человека, плечи и бёдра. Проверьте контур и стороны тела;
маски здесь восстановлены по описанию, так как авторский preprocessing KGI не опубликован.
""")
code('''
prepared = pipeline.preprocess(cloth_path, person_path, garment_type)
from online_wearing.pipeline import point_overlay

views = [
    (point_overlay(prepared.cloth, prepared.cloth_points), "GarmentIQ → KGI-32"),
    (point_overlay(prepared.person, prepared.pose), "OpenPose → KGI-10"),
    (prepared.parse_lip, "SCHP LIP-20"),
    (prepared.parse, "KGI-13"),
    (prepared.parse_agnostic, "Agnostic parse"),
    (prepared.removal, "Removal mask: white = remove"),
]
fig, axes = plt.subplots(2, 3, figsize=(12, 11))
for ax, (image, title) in zip(axes.flat, views):
    options = {"cmap": "tab20", "vmin": 0, "vmax": 19} if "parse" in title.lower() or "LIP" in title or "KGI-13" in title else {}
    ax.imshow(image, **options)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
print(prepared.metadata)
''')
markdown("""
## 6. Одна полная примерка

Установите `RUN_SINGLE=True` в параметрах. Ускоритель выбирается вручную в Kaggle Settings,
затем `DEVICE='cuda:0'` и при необходимости `SCI_PRECISION='float16'`. Повторно создайте класс.
Вызов для своих фотографий: `pipeline(cloth_image, person_image, 'long sleeve top')`.
Ниже используется уже проверенный `prepared`, чтобы не повторять препроцессинг.
""")
code('''
if RUN_SINGLE:
    result = pipeline.infer_prepared(prepared, seed=SEED)
    result_path = result.save(RUN_DIR / "preview", debug=True)
    fig, axes = plt.subplots(1, 5, figsize=(18, 6))
    views = [(Image.open(cloth_path), "Cloth"), (Image.open(person_path), "Person"),
             (result.artifacts["incomplete"], "TPS / SCI input"),
             (result.image, "KGI result"), (Image.open(manifest.parent / record["target"]), "Target: metrics only")]
    for ax, (image, title) in zip(axes, views):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    from online_wearing.evaluation import image_metrics
    print(image_metrics(result.image, manifest.parent / record["target"]))
    print("Inference excluding preprocessing, seconds:", result.metadata["inference_seconds"])
    print("Saved:", result_path)
else:
    print("Full inference is opt-in: set RUN_SINGLE=True after inspecting preprocessing.")
''')
markdown("""
## 7. Полный прогон 1k с возобновлением и отчётом ошибок

Установите `RUN_BATCH=True`. Вызывается настоящая цепочка для всех 1000 записей;
успехи пропускаются при повторном запуске, ошибки пробуются заново. Итог содержит число
успехов/ошибок. MAE/PSNR/SSIM считаются только по успешным примерам, по всему RGB-кадру;
они не измеряют физическую посадку одежды. LPIPS/FID не выдаются без отдельной реализации.

Для продолжения в новой сессии сохраните `runs/kgi_s...` как Kaggle output и прикрепите его,
задав `PREVIOUS_RUN`. Сохранённую выборку можно подключить через `EXISTING_SUBSET`.
Конфигурация sampling/весов и manifest проверяется, чтобы не смешивать разные эксперименты.
""")
code('''
if RUN_BATCH:
    if PREVIOUS_RUN is not None:
        if RUN_DIR.exists() and (RUN_DIR / "metrics.jsonl").exists():
            print("Current run log exists; continuing it without overwriting from PREVIOUS_RUN.")
        else:
            shutil.copytree(PREVIOUS_RUN, RUN_DIR, dirs_exist_ok=True)
    from online_wearing.evaluation import evaluate_subset
    summary = evaluate_subset(pipeline, manifest, RUN_DIR, limit=1000, seed=SEED,
                              save_debug_first=3, fail_fast=False)
    print(json.dumps(summary, indent=2))
else:
    print("Batch is opt-in: set RUN_BATCH=True to process the 1000-record subset.")
''')
markdown("""
## 8. Артефакты

`result.png` — RGB-результат 1024×768; `metadata.json` — конфигурация и точки.
В первых трёх примерах сохраняются исходные фото, карты классов (индексные PNG), маски,
landmark overlays и TPS-композиция. `metrics.jsonl` — журнал, `summary.json` — агрегаты.
Не интерпретируйте тёмную индексную PNG как неудачную маску: для просмотра используйте палитру.

В следующем этапе `measurements_cm` можно использовать для обучения размерного модуля.
В этом baseline они намеренно не изменяют точки или изображение.
""")
code('''
if RUN_DIR.exists():
    shutil.copy2(WORK / "environment.txt", RUN_DIR / "environment.txt")
    shutil.copy2(manifest.parent / "source.json", RUN_DIR / "dataset_source.json")
    if (REPO / ".git").exists():
        commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        (RUN_DIR / "code_commit.txt").write_text(commit + "\\n")
    archive = shutil.make_archive(str(WORK / "onlinewearing_results"), "zip", root_dir=RUN_DIR)
    print("Kaggle output archive:", archive)
else:
    print("No full inference outputs yet; preprocessing inspection is complete.")
''')

for index, cell in enumerate(cells):
    cell["id"] = f"kgi-{index:02d}"
notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
                metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                              language_info=dict(name="python", version="3.11"),
                              kaggle=dict(accelerator="none", isGpuEnabled=False, isInternetEnabled=True,
                                          language="python", sourceType="notebook")))
if __name__ == "__main__":
    path = ROOT / "notebooks/kaggle_kgi_end_to_end.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path)
