"""Convert the frozen UAVSwarmV2 MOT-train temporal detector split (YOLO labels) to COCO json.

The split directory is the one used for the YOLO11s detector (train.txt / val.txt plus
labels/<subset>/<sequence>/<frame>.txt with normalised cx, cy, w, h). Reusing it keeps the
image list and the clipped boxes identical across detectors. Image paths in the json are
relative to the MOT `train` directory: <sequence>/img1/<frame>.jpg.
"""
import argparse
import configparser
import hashlib
import json
from pathlib import Path


def sequence_size(mot_train, name, cache):
    if name not in cache:
        ini = configparser.ConfigParser()
        ini.read(mot_train / name / "seqinfo.ini")
        info = ini["Sequence"]
        cache[name] = (int(info["imWidth"]), int(info["imHeight"]), info.get("imDir", "img1"))
    return cache[name]


def convert(split_dir, mot_train, subset):
    images, annotations, sizes, videos = [], [], {}, {}
    for line in (split_dir / f"{subset}.txt").read_text().split():
        _, _, name, frame_file = Path(line).parts
        width, height, img_dir = sequence_size(mot_train, name, sizes)
        video_id = videos.setdefault(name, len(videos) + 1)
        image_id = len(images) + 1
        images.append({
            "id": image_id, "file_name": f"{name}/{img_dir}/{frame_file}", "width": width, "height": height,
            "frame_id": int(Path(frame_file).stem), "video_id": video_id,
        })
        label = split_dir / "labels" / subset / name / (Path(frame_file).stem + ".txt")
        if not label.exists():
            continue
        for row in label.read_text().splitlines():
            if not row.strip():
                continue
            _, cx, cy, w, h = (float(v) for v in row.split())
            bw, bh = w * width, h * height
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": 1, "iscrowd": 0,
                "bbox": [cx * width - bw / 2, cy * height - bh / 2, bw, bh], "area": bw * bh, "track_id": -1,
            })
    return {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "UAV"}],
            "videos": [{"id": v, "file_name": k} for k, v in videos.items()]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--mot-train", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    summary = {}
    for subset in ("train", "val"):
        data = convert(a.split_dir, a.mot_train, subset)
        path = a.output / f"{subset}.json"
        path.write_text(json.dumps(data))
        summary[subset] = {"images": len(data["images"]), "boxes": len(data["annotations"]),
                           "sequences": len(data["videos"]), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for name in ("train.txt", "val.txt", "split_manifest.json"):
        summary[name + "_sha256"] = hashlib.sha256((a.split_dir / name).read_bytes()).hexdigest()
    (a.output / "conversion_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
