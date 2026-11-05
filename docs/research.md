# Проверенные источники и выбор моделей

Источники просмотрены 5 ноября 2026 года. Реальный inference по этим весам не выполнялся.

| Решение | Первичный источник | Что проверялось |
|---|---|---|
| Сохранить KG/PG/SCI | [KGI](https://github.com/lizhi-ntu/KGI), локальные AGENT.md/README.md/agent и исходники | Три модели, 32/10 точек, 19→13 PG, конфигурация SCI |
| Гарментные точки | [GarmentIQ](https://github.com/lygitdata/GarmentIQ), [PyPI](https://pypi.org/project/garmentiq/), [classes](https://github.com/lygitdata/GarmentIQ/blob/main/src/garmentiq/garment_classes.py) | PoseHighResolutionNet, HRNet веса, срезы классов 25:58 / 0:25 / 128:143 |
| Препроцессинг HRNet | [detection/utils.py](https://github.com/lygitdata/GarmentIQ/blob/main/src/garmentiq/landmark/detection/utils.py) | Affine, RGB ImageNet normalize, heatmap decoder, inverse transform |
| Порядок точек | [DeepFashion2](https://github.com/switchablenorms/DeepFashion2), GarmentIQ instruction JSON, локальный KGI dataset_demo.py | Категорийные схемы; long/vest индексы KGI, экспериментальное расширение short |
| Позы OpenPose | [controlnet_aux/open_pose](https://github.com/huggingface/controlnet_aux/tree/master/src/controlnet_aux/open_pose), [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) | Body выдаёт COCO-18 в пикселях; detector high-level возвращает нормированные точки |
| Парсер SCHP LIP-20 | [IDM-VTON preprocessing](https://github.com/yisol/IDM-VTON/tree/main/preprocess/humanparsing), [ONNX weights](https://huggingface.co/yisol/IDM-VTON/tree/main/humanparsing), [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) | 512 square affine input, BGR normalize, финальные logits, inverse affine |
| Таблица 20→13 | [HR-VITON cp_dataset.py](https://github.com/sangyun884/HR-VITON/blob/main/cp_dataset.py) | Объединение классов, включая left/right и socks |
| Альтернатива FASHN | [FASHN Human Parser](https://github.com/fashn-AI/fashn-human-parser), [labels](https://github.com/fashn-AI/fashn-human-parser/blob/main/src/fashn_human_parser/labels.py) | 18 классов; arms/legs объединяют стороны, что потребовало бы дополнительного восстановления |
| FITVTO-1k | [Dataset card/schema](https://huggingface.co/datasets/Yuanhao-Harry-Wang/fitvto-100k/blob/main/README.md) | Реальные колонки cloth/person/target, eval split, 7 числовых мерок в cm |
| Ограничение загрузки | [HF streaming](https://huggingface.co/docs/datasets/stream) | streaming=True, без full download/shuffle; максимум 1000 экспортируемых записей |

Выбор SCHP вместо FASHN — инженерное решение ради LIP-20 и левой/правой стороны,
а не доказанное превосходство качества. Отдельного сравнения этих моделей не проводилось.
Доступность страницы/списка файлов не означает, что веса были скачаны и проверены.
Схема HF FITVTO указывает `cloth`, хотя короткий Usage-пример в карточке обращается
к `image`; поэтому код проверяет фактическую схему и не угадывает колонку.

GarmentIQ закреплён на опубликованном PyPI `0.0.4.11`. У PyPI README и GitHub main
в доступных снимках различались детали device API; адаптер использует архитектуру,
базовые preprocessing/decode helpers и собственную явную загрузку torch weights.
Полный dependency lock без установки/разрешения окружения не заявляется: версии
транзитивных библиотек согласует pip, фактическое окружение записывается в Kaggle.
