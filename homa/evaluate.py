"""Reuse the pinned evaluator unchanged, including GT-only frames."""
import argparse
import json
from pathlib import Path
import importlib.util


def evaluate(data, split, tracks, output, names):
    source=Path(__file__).resolve().parents[1]/'tools/evaluate_uavswarm_mot.py'
    spec=importlib.util.spec_from_file_location('pinned_eval',source); e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
    e.mm.lap.default_solver='lap'; accum=[]; hota={}
    for name in names:
        gt=data/split/name/'gt/gt.txt'; pred=tracks/f'{name}.txt'
        accum.append(e.mm.utils.compare_to_groundtruth(e.mm.io.loadtxt(str(gt),fmt='mot15-2D',min_confidence=1),e.mm.io.loadtxt(str(pred),fmt='mot15-2D',min_confidence=-1),'iou',distth=.5))
        hota[name]=e.HOTA().eval_sequence(e.trackeval_data(e.read_mot_boxes(gt,1),e.read_mot_boxes(pred,-1)))
    summary=e.mm.metrics.create().compute_many(accum,names=names,metrics=e.MOT_FIELDS,generate_overall=True)
    metric=e.HOTA()
    result={'protocol':{'clear_id_iou':.5,'hota_iou':[.05,.95],'frame_policy':'GT/result union','gt_used_only_for_evaluation':True},'overall':{'motmetrics':e.serialise_mot_row(summary.loc['OVERALL']),'hota':e.serialise_hota_result(metric.combine_sequences(hota),metric)},'per_sequence':{name:{'motmetrics':e.serialise_mot_row(summary.loc[name]),'hota':e.serialise_hota_result(hota[name],metric)} for name in names}}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--split',required=True); p.add_argument('--tracks',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--sequences',nargs='+',required=True)
    a=p.parse_args(); evaluate(a.data,a.split,a.tracks,a.output,a.sequences)

if __name__=='__main__':main()
