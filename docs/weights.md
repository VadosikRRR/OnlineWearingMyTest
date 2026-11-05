# Веса для Kaggle

Все шесть путей принимает `Weights`; ошибки перечисляют все отсутствующие файлы.
В актуальном ноутбуке вручную прикрепляются только KG/PG/SCI; остальные три файла
скачиваются с HF ячейкой весов при `DOWNLOAD_AUX_FROM_HF=True` (по умолчанию).
`DOWNLOAD_AUX_FROM_HF=False` включает полностью локальную схему из таблицы ниже.
Весов в этом репозитории нет. Архитектуры KG/PG/SCI включены в `src/online_wearing/vendor`,
остальные загружаются из установленных библиотек или ONNX. Клонировать репозитории моделей не нужно.

| Поле | Файл в Kaggle Dataset `onlinewearing-weights` | Источник |
|---|---|---|
| `kg` | `kg/step_299999.pt` | [Архив KG KGI](https://drive.google.com/file/d/1FQbeWkcqgSqycznGW6INWt2kkrk5nG3H/view) |
| `pg` | `pg/step_9999.pt` | [Архив PG KGI](https://drive.google.com/file/d/1_2lTeLdCczBnnaXsMxBtvbLZv2GY3PCp/view) |
| `sci` | `sci/ema_0.9999_300000.pt` | [Архив SCI KGI](https://drive.google.com/file/d/1guxm63mRMH64xiycDHsEqNb_q3KqueDB/view) |
| `garment_hrnet` | `hrnet.pth` | [lygitdata/garmentiq](https://huggingface.co/lygitdata/garmentiq/blob/main/hrnet.pth) |
| `openpose_body` | `body_pose_model.pth` | [lllyasviel/Annotators](https://huggingface.co/lllyasviel/Annotators/blob/main/body_pose_model.pth) |
| `human_parser_lip` | `parsing_lip.onnx` | [yisol/IDM-VTON](https://huggingface.co/yisol/IDM-VTON/blob/main/humanparsing/parsing_lip.onnx) |

Ссылки KG/PG/SCI перенесены из исходного README KGI; скачивание Google Drive и
содержимое весов в этой работе не проверялись. Архивы нужно распаковать и найти
указанные файлы, а не передавать ZIP вместо checkpoint.
Не подменяйте LIP-20 на `parsing_atr.onnx`: отличаются классы и их порядок.
Не нужны hand/face OpenPose weights, DensePose, CLIP, VAE или веса Stable Diffusion.

Структура полностью локального Dataset (в режиме HF последние три файла здесь не обязательны):

```text
/kaggle/input/onlinewearing-weights/
  kg/step_299999.pt
  pg/step_9999.pt
  sci/ema_0.9999_300000.pt
  hrnet.pth
  body_pose_model.pth
  parsing_lip.onnx
```

Реальный mount Kaggle может отличаться: исправьте `WEIGHTS_ROOT` в первой ячейке.
Три вспомогательных файла можно получить **в отдельной подготовительной Kaggle-сессии**:

```bash
python scripts/download_aux_weights.py /kaggle/working/aux_weights
```

Помощник вызывает HF API/download для конкретных файлов, фиксирует commit каждого
источника в `sources.json`, не клонирует репозитории и не скачивает основной IDM-VTON.
Затем объедините эти файлы с тремя KGI checkpoints в одном Kaggle Dataset.

Загрузка torch checkpoints использует `weights_only=True`, `map_location="cpu"`
и строгую проверку ключей. Поддержаны `state_dict`, `model_state_dict`, `model` wrappers
и префикс `module.`. Несовпадение архитектуры/весов вызывает ошибку; частично загруженная
модель с пропущенными параметрами не запускается.

Лицензии кода, датасета и весов различаются; источники перечислены в
[THIRD_PARTY.md](../THIRD_PARTY.md). FITVTO опубликован под CC BY-NC-ND 4.0,
IDM-VTON — CC BY-NC-SA 4.0; у OpenPose свои условия некоммерческого использования.
