# traffic-sign-edge-ai
Traffic Sign Detection and INT8 Quantization on Raspberry Pi 5

## Dataset preparation (GTSDB → YOLO)

`src/prepare_dataset.py` converts the official GTSDB `FullIJCNN2013` release into a YOLO detection dataset under `data/gtsdb/`. Raw files in `data/FullIJCNN2013/` are read-only inputs and are never modified.

### Usage

```bash
python3 src/prepare_dataset.py
```

Explicit example:

```bash
python3 src/prepare_dataset.py \
  --source data/FullIJCNN2013 \
  --output data/gtsdb \
  --seed 42 \
  --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
```

If the raw dataset is unavailable, the script checks local ZIP archives (`data/FullIJCNN2013.zip`, `data/GTSDB_FullIJCNN2013.zip`) before printing manual instructions. Downloads are opt-in:

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

- `--split-mode random` (default): sorted image names are shuffled with `random.Random(seed)`, then divided using the configured ratios. An image appears in exactly one split.
- `--split-mode official`: preserves the official test range `00600–00899`; the remaining images are split into train/val from the same seed. Official test membership was verified by SHA-256 checks at sampled positions (`00000→00600`, `00001→00601`, `00150→00750`, `00299→00899`) against the official `TestIJCNN2013.zip`.
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
| **Horizontal flip (`fliplr`)** | `0