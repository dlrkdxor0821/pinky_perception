# Labeled test set for mAP evaluation

`eval/eval_map.py` needs a dataset in **YOLO format** with ground-truth labels.
When your test labels are ready, lay them out like this and point `--data` at
the yaml.

## Layout

```
eval/datasets/test/
├── images/            # test images
│   ├── 000001.jpg
│   └── ...
├── labels/            # one .txt per image, same basename
│   ├── 000001.txt     # each line: "<class> <cx> <cy> <w> <h>"  (normalized 0..1)
│   └── ...
└── data.yaml
```

## data.yaml

```yaml
path: eval/datasets/test     # dataset root (relative to where you run, or absolute)
val: images                  # images used for evaluation (eval_map.py --split val)
names:                       # class id -> name (MUST match your training classes)
  0: person
  1: bottle
  2: chair
```

## Notes

- Use the **same class set / model** on edge and server so the mAP numbers are
  comparable. Only the runtime differs (NCNN-light vs PyTorch-full).
- Label files are normalized: `cx, cy, w, h` are fractions of image width/height.
- This dataset is git-ignored by default (see repo `.gitignore` if you want to
  track a small fixed test set).
- For a public sanity check you can use the COCO `val2017` split with the
  standard `coco.yaml`; Ultralytics will download it on first `val()`.

## Run

```bash
python3 eval/eval_map.py --model models/yolo11n_ncnn_model --data eval/datasets/test/data.yaml --label edge
python3 eval/eval_map.py --model yolo11n.pt                 --data eval/datasets/test/data.yaml --label server
python3 eval/compare_map.py benchmark/results/map_edge.json benchmark/results/map_server.json
```
