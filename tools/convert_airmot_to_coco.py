"""Convert AIRMOT MOT-format training sequences to COCO json with a per-sequence temporal split.

Reads <airmot-train>/<sequence>/seqinfo.ini and gt/gt.txt (frame,id,x,y,w,h,conf,class,vis)
only; the official test directory is never opened. For a sequence of L frames the last
floor(L * val_tail) frames go to val and the frames before them to train. Image paths in
the json are relative to the AIRMOT `images/train` directory: <sequence>/img1/<frame>.jpg.
"""
import argparse
import configparser
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sequence(seq):
    ini = configparser.ConfigParser()
    ini.read(seq / "seqinfo.ini")
    info = ini["Sequence"]
    meta = {"length": int(info["seqLength"]), "width": int(info["imWidth"]), "height": int(info["imHeight"]),
            "img_dir": info.get("imDir", "img1"), "ext": info.get("imExt", ".jpg")}
    boxes = {}
    for row in (seq / "gt" / "gt.txt").read_text().split():
        frame, track, x, y, w, h = (float(v) for v in row.split(",")[:6])
        assert 1 <= frame <= meta["length"] and w > 0 and h > 0, (seq.name, row)
        assert x >= 0 and y >= 0 and x + w <= meta["width"] and y + h <= meta["height"], (seq.name, row)
        boxes.setdefault(int(frame), []).append((int(track), x, y, w, h))
    return meta, boxes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--airmot-train", type=Path, required=True, help="AIRMOT images/train directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--val-tail", type=float, default=0.2)
    p.add_argument("--sequences", nargs="+")
    a = p.parse_args()

    names = a.sequences or sorted(x.name for x in a.airmot_train.iterdir() if (x / "seqinfo.ini").is_file())
    subsets = {s: {"images": [], "annotations": []} for s in ("train", "val")}
    summary = {"source": "AIRMOT images/train", "val_tail": a.val_tail, "sequences": {}}
    for video_id, name in enumerate(names, 1):
        meta, boxes = read_sequence(a.airmot_train / name)
        first_val = meta["length"] - int(meta["length"] * a.val_tail) + 1
        counts = {"train": [0, 0], "val": [0, 0]}
        for frame in range(1, meta["length"] + 1):
            subset = "train" if frame < first_val else "val"
            images, annotations = subsets[subset]["images"], subsets[subset]["annotations"]
            image_id = len(images) + 1
            images.append({
                "id": image_id, "file_name": f"{name}/{meta['img_dir']}/{frame:06d}{meta['ext']}",
                "width": meta["width"], "height": meta["height"], "frame_id": frame, "video_id": video_id,
            })
            for track, x, y, w, h in sorted(boxes.get(frame, [])):
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id, "category_id": 1, "iscrowd": 0,
                    "bbox": [x, y, w, h], "area": w * h, "track_id": track,
                })
            counts[subset][0] += 1
            counts[subset][1] += len(boxes.get(frame, []))
        summary["sequences"][name] = {
            "frames": meta["length"], "train_frames": [1, first_val - 1], "val_frames": [first_val, meta["length"]],
            "train_images": counts["train"][0], "train_boxes": counts["train"][1],
            "val_images": counts["val"][0], "val_boxes": counts["val"][1],
            "gt_sha256": sha256(a.airmot_train / name / "gt" / "gt.txt"),
        }

    a.output.mkdir(parents=True, exist_ok=False)
    videos = [{"id": i, "file_name": name} for i, name in enumerate(names, 1)]
    for subset, data in subsets.items():
        data.update({"categories": [{"id": 1, "name": "UAV"}], "videos": videos})
        path = a.output / f"{subset}.json"
        path.write_text(json.dumps(data))
        summary[subset] = {"images": len(data["images"]), "boxes": len(data["annotations"]), "sha256": sha256(path)}
    (a.output / "conversion_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
