# DeepD3 RESPAN Fine-Tuning

Run all commands from `/home/nochen/code/DeepD3`.

```bash
conda run -n deepd3 python scripts/deepd3_finetune/prepare_data.py --overwrite
conda run -n deepd3 python scripts/deepd3_finetune/convert_to_d3set.py --overwrite
conda run -n deepd3 python scripts/deepd3_finetune/download_models.py --models DeepD3_8F.h5 DeepD3_16F.h5
bash scripts/deepd3_finetune/run_deepd3_gpu.sh scripts/deepd3_finetune/gpu_preflight.py
bash scripts/deepd3_finetune/run_deepd3_gpu.sh scripts/deepd3_finetune/smoke_check.py --train-batch
bash scripts/deepd3_finetune/run_deepd3_gpu.sh scripts/deepd3_finetune/train_deepd3_finetune.py --dataset invivo
bash scripts/deepd3_finetune/run_deepd3_gpu.sh scripts/deepd3_finetune/train_deepd3_finetune.py --dataset confocal
bash scripts/deepd3_finetune/run_deepd3_gpu.sh scripts/deepd3_finetune/evaluate_holdout.py --dataset invivo
bash scripts/deepd3_finetune/run_deepd3_gpu.sh scripts/deepd3_finetune/evaluate_holdout.py --dataset confocal
conda run -n deepd3 python scripts/deepd3_finetune/render_respan_style_holdout.py --dataset all
```

The default training base is `models/pretrained/DeepD3_16F.h5` with mixed/native resolution
sampling (`target_resolution=None`). Use `DeepD3_8F.h5` for quick smoke tests.

Heldout evaluation uses tiled inference by default (`--tile-size 512 --tile-overlap 64`) to fit
the RTX 5060 8 GB GPU. Outputs are written to
`models/respan_finetuned/{invivo,confocal}/heldout_eval/`.

RESPAN-style overlays and metrics are written under
`models/respan_finetuned/{invivo,confocal}/evaluation/`.

The wrapper is needed because TensorFlow 2.21's CUDA pip wheels place libraries under
`site-packages/nvidia/.../lib`, which must be present on `LD_LIBRARY_PATH` before Python starts.
