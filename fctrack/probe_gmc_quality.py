"""Measure sparse-flow support for frozen GMC transforms without changing them.

Image/corner/LK settings follow Exp055. This measures observed flow disagreement,
not calibrated covariance or true registration error. No GT is read. Identity
warps can represent either static estimates or earlier estimator failure.
"""
import argparse
import configparser
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def disagreement(previous, moved, backward, warp):
    """Nominal original pixels under the legacy downscale=2 convention.

    Odd image dimensions are rounded down on resize, as in Exp055; multiplying
    coordinates by two preserves that legacy convention, not exact rescaling.
    """
    predicted = previous @ warp[:, :2].T + warp[:, 2]
    residual = np.linalg.norm(predicted - moved, axis=1)
    fb = np.linalg.norm(previous - backward, axis=1)
    return {"median_residual_px": float(np.median(residual)),
            "p90_residual_px": float(np.percentile(residual, 90)),
            "fb_median_px": float(np.median(fb)),
            "fraction_within_6px": float(np.mean(residual <= 6))}


def process(root, cfg, output, limit=0):
    cell = cfg['cells']['v1_dev_last']
    data = root / cell['dataset_root'] / cell['split']
    detroot = root / cell['detections']
    gmcroot = root / cfg['probe']['gmc']
    names = sorted(x.name for x in detroot.glob('UAVSwarm-*') if x.is_dir())
    names = names[:limit] if limit else names
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(4)
    summary = {}
    for name in names:
        seq = data / name
        ini = configparser.ConfigParser()
        ini.read(seq / 'seqinfo.ini')
        info = ini['Sequence']
        length, width, height = (int(info[k]) for k in ('seqLength', 'imWidth', 'imHeight'))
        ext = info.get('imExt', '.jpg')
        det = np.loadtxt(detroot / name / 'det.txt', delimiter=',', ndmin=2)
        frozen = np.loadtxt(gmcroot / (name + '.txt'), ndmin=2)
        warps = {int(row[0]): row[1:].reshape(2, 3) for row in frozen}
        if set(warps) != set(range(2, length + 1)):
            raise ValueError('Incomplete frozen GMC: ' + name)

        def read(frame):
            image = cv2.imread(str(seq / 'img1' / f'{frame:06d}{ext}'), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError((name, frame))
            image = cv2.resize(image, (int(width / 2), int(height / 2)))
            mask = np.full(image.shape, 255, np.uint8)
            for row in det[(det[:, 0] == frame) & (det[:, 6] >= .1)]:
                x, y, w, h = row[2:6]
                x1, y1 = int((x - w / 2) / 2), int((y - h / 2) / 2)
                x2, y2 = int(np.ceil((x + 1.5 * w) / 2)), int(np.ceil((y + 1.5 * h) / 2))
                mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0
            return image, mask

        previous, previous_mask = read(1)
        rows = []
        for frame in range(2, length + 1):
            current, current_mask = read(frame)
            warp = warps[frame]
            identity = bool(np.array_equal(warp, np.eye(3)[:2]))
            row = {'sequence': name, 'frame': frame, 'corner_count': 0, 'valid_count': 0,
                   'median_residual_px': None, 'p90_residual_px': None, 'fb_median_px': None,
                   'fraction_within_6px': None, 'identity_fallback_ambiguous': identity,
                   'quality_observed': False}
            corners = cv2.goodFeaturesToTrack(previous, maxCorners=1000, qualityLevel=.01,
                                              minDistance=1, blockSize=3, mask=previous_mask)
            if corners is not None:
                row['corner_count'] = len(corners)
            if corners is not None and len(corners) >= 10:
                moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, corners, None)
                if moved is not None and status is not None:
                    back, backstatus, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None)
                    if back is not None and backstatus is not None:
                        p, m, b = corners.reshape(-1, 2), moved.reshape(-1, 2), back.reshape(-1, 2)
                        ok = (status.ravel() == 1) & (backstatus.ravel() == 1)
                        ok &= np.isfinite(p).all(1) & np.isfinite(m).all(1) & np.isfinite(b).all(1)
                        row['valid_count'] = int(ok.sum())
                        if ok.any():
                            row.update(disagreement(p[ok] * 2, m[ok] * 2, b[ok] * 2, warp))
                            row['quality_observed'] = True
            rows.append(row)
            previous, previous_mask = current, current_mask
        (output / (name + '.jsonl')).write_text(''.join(json.dumps(x, allow_nan=False) + '\n' for x in rows))
        summary[name] = {'frames': length, 'quality_unobserved': sum(not x['quality_observed'] for x in rows),
                         'identity_ambiguous_frames': sum(x['identity_fallback_ambiguous'] for x in rows),
                         'warp_sha256': digest(gmcroot / (name + '.txt'))}
        print(name, json.dumps(summary[name]), flush=True)
    (output / 'summary.json').write_text(json.dumps({'scope': 'flow disagreement with unchanged frozen GMC, not covariance; nominal downscale=2 pixels; observed does not imply reliable',
                                                    'sequences': summary}, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    process(a.root.resolve(), json.loads(a.config.read_text()), a.output, a.limit)
