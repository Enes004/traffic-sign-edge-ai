# YOLO26n Traffic Sign Detection and Edge Deployment

BİL458 Computer Vision research project. The main experiment compares FP32 and static INT8 ONNX on a Raspberry Pi 5 at 640 px. A separately trained 320 px configuration measures the resolution trade-off. The held-out GTSDB split has 227 images and three target classes.

[Final experiment summary](docs/final_experiment_summary.md) · [Notebook](notebooks/traffic_sign_edge_ai.ipynb) · [Local CPU demo](results/demo/yolo26n_img640_seed0_fp32.mp4) · [Experiment history](archive.md)

## Installation

From the repository root, use Python 3.10+ in an isolated environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

FFmpeg with the `libx264` encoder is required to render the MP4. On Debian/Ubuntu, install the `ffmpeg` system package. Training, export and quantization additionally use `ultralytics==8.4.215`, `onnx==1.23.2` and `onnxslim==0.1.99`. Install a CUDA-enabled PyTorch build for training on an NVIDIA GPU. The original training environment used PyTorch 2.9.1+cu128.

For Pi runtime only, install `numpy==2.2.6 opencv-python-headless==4.12.0.88 onnxruntime==1.23.2 psutil` with pip in a separate environment. Use a 64-bit Pi OS and the matching Python wheels. Dataset preparation uses Pillow. The saved Pi results can be summarized on a workstation.


## Dataset preparation (GTSDB → YOLO)

`src/prepare_dataset.py` converts the official GTSDB `FullIJCNN2013` release into a YOLO detection dataset under `data/gtsdb/`. Raw files in `data/raw/FullIJCNN2013/` are read-only inputs and are never modified.

### Usage

```bash
python3 src/prepare_dataset.py
```

Explicit example:

```bash
python3 src/prepare_dataset.py \
  --source data/raw/FullIJCNN2013 \
  --output data/gtsdb \
  --split-mode official \
  --seed 42 \
  --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
```

If the raw dataset is unavailable, `setup.sh` or `src/prepare_dataset.py --download` fetches the official archive into `data/raw/`. The script checks local ZIP archives (`data/raw/FullIJCNN2013.zip`, `data/raw/GTSDB_FullIJCNN2013.zip`) before printing manual instructions. Downloads are opt-in:

```bash
python3 src/prepare_dataset.py --download
```

The verified official archive is `https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip` (1,661,866,983 bytes). The archive publisher does not provide checksums, so the script verifies the size and ZIP structure.

### Output layout

```text
data/gtsdb/
├── dataset.yaml
├── images/{train,val,test}/NNNNN.jpg
├── labels/{train,val,test}/NNNNN.txt
├── preparation_report.json
└── .prepare_manifest.json
```

`dataset.yaml` uses the three official detection categories as YOLO IDs `0`, `1`, `2`; its `path` entry is the absolute path to the output directory.

### Official mapping and excluded classes

The mapping is transcribed from the `ReadMe.txt` bundled with `FullIJCNN2013` and from the official benchmark code `TSD_readGTData.m`; both are documented at <https://github.com/houbensebastian/GermanTrafficSignBenchmarks>.

- YOLO `0` — `prohibitory` (also called “prohibitive”): source classes `0,1,2,3,4,5,7,8,9,10,15,16`
- YOLO `1` — `danger`: source classes `11,18,19,20,21,22,23,24,25,26,27,28,29,30,31`
- YOLO `2` — `mandatory`: source classes `33,34,35,36,37,38,39,40`
- Excluded `other`: source classes `6,12,13,14,17,32,41,42`

Policy: if an image contains **any** annotation from the excluded `other` set, the entire image is omitted from the YOLO dataset. This prevents excluded annotated signs from silently becoming background labels. No excluded class is ever mapped to a YOLO class or to background. The report lists the dropped image names and counts.

### Empty images, duplicates, and validation

- The official `gt.txt` contains 1,213 rows and 1,212 unique annotations. One exact duplicate (`00340.ppm`, class 14) is present; it is skipped with an explicit warning, and the raw row count is preserved in the report.
- 159 images have no `gt.txt` rows. They are included with empty label files by default and reported as unannotated negatives. `--exclude-unannotated` omits them instead. They are structurally validated but not independently visually audited.
- Per-class crop directories are cross-checked against the raw annotation counts.
- Validation checks image counts, missing images/labels, bbox boundaries, normalized coordinates, class distributions, empty labels, split disjointness, and split overlap. Results are printed and written to `preparation_report.json`.

### Splits and reproducibility

- `--split-mode official` (default): preserves the official test range `00600–00899`; the remaining images are split into train/val from the same seed. Official test membership was verified by SHA-256 checks at sampled positions (`00000→00600`, `00001→00601`, `00150→00750`, `00299→00899`) against the official `TestIJCNN2013.zip`.
- `--split-mode random`: sorted image names are shuffled with `random.Random(seed)`, then divided using the configured ratios. An image appears in exactly one split.
- The official competition used 600 train / 300 test. Because images with excluded-class annotations are dropped, prepared counts differ from the official 600/300.
- GTSDB is video-derived and provides no sequence IDs, so random splitting can place near-duplicate frames in different splits. Official mode is closer to the benchmark protocol.
- Same source, seed, split mode, ratios, and JPEG quality produce the same split and labels.

Reruns are idempotent: unchanged JPGs are reused via `.prepare_manifest.json`, labels/YAML/report files are rewritten only when content changes, and stale tracked outputs are removed after a seed or ratio change. Source PPMs and `gt.txt` are never deleted or overwritten.

Coordinates follow the official continuous convention (`width = xmax - xmin`) and are written as normalized `class_id x_center y_center width height` with six decimals. JPEG output defaults to quality 95 with 4:4:4 subsampling.

### Visualize labels

`src/visualize_labels.py` draws class-labelled bounding boxes on 12 annotated images sampled deterministically from the prepared train/val/test sets and writes `results/dataset_validation.jpg`. It only reads the dataset.

```bash
python3 src/visualize_labels.py
```

Optional flags: `--dataset`, `--output`, `--seed` (default `42`), and `--count` (default `12`).

### Dependencies and tests

Required: Python 3.8+ and Pillow. PyYAML is not required. The script never installs packages.

```bash
python3 -m pytest tests/test_prepare_dataset.py -q
```

---

## Model training (YOLO26n — FP32 Baseline)

`src/train.py` trains `YOLO26n` in **FP32** precision on the prepared GTSDB dataset (`--split-mode official`) using deterministic settings across three random seeds (`0, 1, 2`). Best checkpoints are saved to `models/yolo26n_img640_seed{0,1,2}.pt` and training logs to `runs/train/yolo26n_img640_seed{0,1,2}/results.csv`.

### Usage

```bash
python3 src/train.py --imgsz 640 --seed 0
python3 src/train.py --imgsz 640 --seed 1
python3 src/train.py --imgsz 640 --seed 2
```

### Training configuration

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Model** | `YOLO26n` (`yolo26n.pt`) | 121 layers, 2,390,221 params, 5.4 GFLOPs, 5.4 MB |
| **Precision** | `FP32` | Baseline full-precision training |
| **Input size (`--imgsz`)** | `640` (`640×640`) | Fixed resolution for baseline runs |
| **Seeds (`--seed`)** | `0, 1, 2` | `deterministic=True` for reproducibility |
| **Epochs (`--epochs`)** | `100` | Early stopping with `--patience 25` |
| **Batch size (`--batch`)** | `16` | Default batch size |
| **Horizontal flip (`fliplr`)** | `0.0` | Disabled to prevent mirroring traffic sign numbers/symbols |
| **Dataset split** | `official` | Train: 378 images · Val: 81 images (104 instances) · Test: 227 images (held out) |
| **Hardware / Software** | NVIDIA GeForce RTX 5060 Laptop GPU | Python 3.10.12, PyTorch 2.9.1+cu128, Ultralytics 8.4.215 |

### Validation results (3-seed FP32 baseline, `imgsz=640`)

Evaluated on the validation split (`81` images, `104` instances: `68` prohibitory, `25` danger, `11` mandatory). The test split (`227` images) remains untouched during training, calibration, and model selection.

| Seed | Best Epoch | Precision | Recall | mAP50 | mAP50-95 | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 40 / 65 | 0.9615 | 0.8678 | 0.9498 | 0.7727 | `models/yolo26n_img640_seed0.pt` |
| **1** | 100 / 100 | 0.9171 | 0.8788 | 0.9270 | 0.7755 | `models/yolo26n_img640_seed1.pt` |
| **2** | 51 / 76 | 0.9287 | 0.9415 | 0.9456 | 0.7818 | `models/yolo26n_img640_seed2.pt` |
| **Mean ± Std** | — | **0.9358 ± 0.0188** | **0.8960 ± 0.0325** | **0.9408 ± 0.0099** | **0.7767 ± 0.0038** | — |

---

## Resolution ablation (`YOLO26n` — `320×320`, FP32 & Static INT8 PTQ)

To isolate the effect of input resolution on accuracy, quantization sensitivity, and Raspberry Pi 5 CPU latency, `YOLO26n` is also fine-tuned from `yolo26n.pt` at `320×320` on the exact same `official` split (`378 / 81 / 227`) with identical hyperparameters (`epochs=100`, `patience=25`, `batch=16`, `deterministic=True`, `fliplr=0.0`, seeds `0, 1, 2`).

### Training, export, quantization, and verification commands

```bash
for s in 0 1 2; do
  python3 src/train.py --imgsz 320 --seed $s
  python3 src/export_onnx.py --imgsz 320 --seed $s
  python3 src/verify_onnx.py --imgsz 320 --seed $s
  python3 src/quantize_int8.py --imgsz 320 --seed $s
done

python3 src/verify_onnx_suite.py --imgsz 320 --seeds 0 1 2 --eval-val --output results/ablation_320/summary.json
```

### Training validation results (`imgsz=320`, Ultralytics default `rect=True`)

| Seed | Best Epoch | Precision | Recall | mAP50 | mAP50-95 | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 86 / 100 | 0.8836 | 0.6405 | 0.7389 | 0.5241 | `models/yolo26n_img320_seed0.pt` |
| **1** | 83 / 100 | 0.9400 | 0.5692 | 0.6937 | 0.4917 | `models/yolo26n_img320_seed1.pt` |
| **2** | 83 / 100 | 0.8910 | 0.6809 | 0.7710 | 0.5197 | `models/yolo26n_img320_seed2.pt` |
| **Mean ± Std** | — | **0.9049 ± 0.0250** | **0.6302 ± 0.0462** | **0.7345 ± 0.0317** | **0.5118 ± 0.0143** | — |

### Exported ONNX models and `val` pre-check (`imgsz=320`, `batch=1`, `rect=False`, CPU)

- **Input / Output shapes:** `[1, 3, 320, 320]` → `[1, 7, 2100]` (`1.3 GFLOPs` vs. `5.4 GFLOPs` at `640×640`).
- **INT8 PTQ settings:** 100 deterministic `train` calibration images (`calib_seed=42`), QDQ format, `QUInt8` activations, `QInt8` per-channel weights, `MinMax`, `op_types_to_quantize=["Conv"]` (`102` Conv nodes, `102` `INT32` biases, `194` `QuantizeLinear`, `398` `DequantizeLinear`).
- **Model sizes:** FP32 `9.28 MiB` → INT8 `2.75 MiB` (`-70.4%`). All 6 ONNX files passed `onnx.checker`, ONNX Runtime CPU shape/smoke inference checks (`00004.jpg`), and `.pt` vs. FP32 parity (`max mAP delta = 0.0000`, `PASS`).

| Seed | Format | Size | Precision | Recall | mAP50 | mAP50-95 | Model File |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | FP32 | 9.28 MiB | 0.9075 | 0.5896 | 0.6938 | 0.5035 | `models/yolo26n_img320_seed0_fp32.onnx` |
| **0** | INT8 | 2.75 MiB | 0.8949 | 0.5725 | 0.6732 | 0.4785 | `models/yolo26n_img320_seed0_int8.onnx` |
| **1** | FP32 | 9.28 MiB | 0.9240 | 0.5695 | 0.6962 | 0.4884 | `models/yolo26n_img320_seed1_fp32.onnx` |
| **1** | INT8 | 2.75 MiB | 0.9314 | 0.5559 | 0.6644 | 0.4644 | `models/yolo26n_img320_seed1_int8.onnx` |
| **2** | FP32 | 9.28 MiB | 0.8530 | 0.6838 | 0.7516 | 0.5300 | `models/yolo26n_img320_seed2_fp32.onnx` |
| **2** | INT8 | 2.75 MiB | 0.9597 | 0.5987 | 0.7044 | 0.4773 | `models/yolo26n_img320_seed2_int8.onnx` |
| **Mean ± Std** | **FP32** | **9.28 MiB** | **0.8949 ± 0.0304** | **0.6143 ± 0.0498** | **0.7138 ± 0.0267** | **0.5073 ± 0.0172** | — |
| **Mean ± Std** | **INT8** | **2.75 MiB** | **0.9286 ± 0.0265** | **0.5757 ± 0.0176** | **0.6807 ± 0.0172** | **0.4734 ± 0.0064** | — |

Full machine-readable verification metadata (hashes, shapes, graph counts, and validation metrics) is stored in `results/ablation_320/summary.json`, and detailed methodology/results are tracked in `archive.md`.

---

## Final experiment and local reproduction

`models/` contains the six trained checkpoints and 12 ONNX files. `runs/train/` contains training configuration and validation histories. `data/` is prepared locally. `results/final/` contains the ABBA summaries and camera-ready plots. `results/failure_analysis/` contains four real test cases with the underlying detections and matches. Previous benchmark files and the experiment history are retained.

### Rebuild the analysis from saved artifacts

After preparing the dataset with the official split (seed 42), run these commands from the repository root. The first command writes a separate verification copy of the final results. Failure analysis uses seed 0 for all four models. The demo makes ten ONNX predictions on the local CPU and shows each still for three seconds.

```bash
python src/summarize_benchmark.py --final-dir results/reproduction/final
python src/failure_analysis.py
python src/demo_inference.py
python -m pytest -q
python -m notebook notebooks/traffic_sign_edge_ai.ipynb
```

Use **Restart Kernel and Run All Cells** to regenerate the notebook tables and plots from saved files. For headless execution, run the following after creating `results/reproduction/`.

```bash
python -m nbconvert --to notebook --execute --output-dir results/reproduction notebooks/traffic_sign_edge_ai.ipynb
```

The two source logs are `results/abba_3seed_20261123_014946.log` (640 px) and `results/abba_320_20261124_223345.log` (320 px). Their `Saved:` lines define which raw CSV/JSON pairs enter the final analysis. Files from older runs and smoke tests remain outside that inventory. The raw CSV/JSON files hold run-level summaries. Individual inference timings are not included in those files.

### Recreate model artifacts in a separate checkout

The published experiment uses the included artifacts. Execute the following only in a separate checkout if recreating the training/export pipeline, since export and quantization write the canonical model filenames. Calibration uses 100 training images selected with seed 42.

```bash
python -m pip install ultralytics==8.4.215 onnx==1.23.2 onnxslim==0.1.99
for size in 640 320; do
  for seed in 0 1 2; do
    python src/train.py --imgsz "$size" --seed "$seed" --epochs 100 --batch 16 --patience 25 --device 0
    python src/export_onnx.py --imgsz "$size" --seed "$seed" --opset 17
    python src/quantize_int8.py --imgsz "$size" --seed "$seed" --calib-count 100 --calib-seed 42
  done
done
```

### Collect a new Pi campaign

Run on the Pi with the published model files and prepared test split. For the original protocol, set and record the `performance` CPU governor and record `vcgencmd measure_clock arm`, `measure_temp` and `get_throttled` before and after each run. Keep each campaign in a separate root containing `models/` and `data/gtsdb/`, as `--root` also sets the destination for raw results.

```bash
# Set CAMPAIGN_ROOT to the separate Pi campaign directory.
for size in 640 320; do
  for seed in 0 1 2; do
    for variant in fp32 int8 int8 fp32; do
      vcgencmd measure_clock arm
      vcgencmd measure_temp
      vcgencmd get_throttled
      python src/benchmark_pi_v3.py --root "$CAMPAIGN_ROOT" --imgsz "$size" \
        --seed "$seed" --variant "$variant" --threads 4 --warmup 5 --repeats 5 \
        --conf 0.001 --nms-iou 0.7 --max-det 300
      vcgencmd measure_clock arm
      vcgencmd measure_temp
      vcgencmd get_throttled
    done
  done
done 2>&1 | tee "$CAMPAIGN_ROOT/abba_reproduction.log"
```

For a smoke run add `--max-images 5` to a single benchmark invocation. The existing summarizer validates the two dated published campaigns. See the final summary for how their metrics were aggregated. To rerun just the 640 px object-size analysis, use:

```bash
python src/analyze_object_size.py --seeds 0 1 2 --threads 4 --warmup 1 \
  --conf 0.25 --match-iou 0.5 --nms-iou 0.7 --max-det 300 \
  --output-dir results/reproduction/object_size
```
