# Происхождение кода и весов

Основной источник: локальный `../codes/KGI`, upstream
[lizhi-ntu/KGI](https://github.com/lizhi-ntu/KGI), HEAD checkout
`22c65d56650e7d7943582355cd39360f6166e089`.
Исходные файлы пользователя не изменялись. Не переносились training loops,
датасеты, кеши `.pyc` и ненужные копии guided_diffusion.

| Включённый файл | Исходник |
|---|---|
| `src/online_wearing/vendor/kg.py` | `codes_kg/models/semgcn.py` |
| `src/online_wearing/vendor/pg.py` | `codes_pg/networks_pg.py` |
| `src/online_wearing/vendor/sci/{unet,nn,gaussian_diffusion,respace,script_util,scheduler}.py` | `codes_demo/guided_diffusion/` |
| `src/online_wearing/vendor/sci/fp16_util.py` | Два inference conversion helpers из `codes_demo/guided_diffusion/fp16_util.py` |
| `src/online_wearing/conditioning.py` | `draw_skeleton`, `draw_cloth` из `generate_demo.py` |

Адаптации: nonpersistent buffers KG, device-safe PG one-hot helper, удаление
standalone CUDA demo из PG, обход
autograd checkpoint wrapper в `no_grad`, исключение training/logger части fp16 util,
замена torchvision ToTensor в rasterization на эквивалентное torch-преобразование.
Архитектуры и имена параметров сохранены. TPS и маски реализованы отдельно;
отличия описаны в `docs/architecture.md`. Контрольные суммы включённых файлов
записаны в `docs/vendor_manifest.json`.

KGI основан на [SemGCN](https://github.com/garyzhao/SemGCN),
[semantic-diffusion-model](https://github.com/WeilunWang/semantic-diffusion-model),
[RePaint](https://github.com/andreas128/RePaint),
[OpenAI guided-diffusion](https://github.com/openai/guided-diffusion).
Ссылки/авторские комментарии в исходниках сохранены. В локальном KGI нет LICENSE;
общая новая лицензия для заимствованного кода здесь не назначается.

GarmentIQ и controlnet-aux подключаются как библиотеки. SCHP ONNX берётся из
IDM-VTON. Весовые файлы не включены, список источников: `docs/weights.md`.
Условия их использования определяются соответствующими авторами/репозиториями.
Схема/препроцессинг SCHP сверялись с IDM-VTON и SCHP; таблица LIP→13 — с HR-VITON.

Ссылка на исходную работу KGI:

```bibtex
@InProceedings{Li_2023_ICCV,
  author = {Li, Zhi and Wei, Pengfei and Yin, Xiang and Ma, Zejun and Kot, Alex C.},
  title = {Virtual Try-On with Pose-Garment Keypoints Guided Inpainting},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2023},
  pages = {22788--22797}
}
```
