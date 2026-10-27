# Extracted without algebra changes from pinned TraDeS, see LICENSE.
import numpy as np
import math
from .image import gaussian_radius, draw_umich_gaussian_1d

def get_cost_volume_label(self, bbox, k, ret, pre_ct):
  pre_ct = pre_ct / 2
  pre_ct_int = pre_ct.astype(np.int32)
  bbox = bbox / 2
  h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
  radius = gaussian_radius((math.ceil(h), math.ceil(w)))
  radius = max(0, int(radius))
  ct = np.array(
    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
  ct_int = ct.astype(np.int32)

  # 1D cost_volume
  cls_id = int(ct_int[1] * (self.opt.output_w / 2) + ct_int[0])
  ret['h_hm_down'][cls_id] = 0
  ret['w_hm_down'][cls_id] = 0
  ret['cat_down'][k] = cls_id
  ret['mask_down'][k] = 1
  ret['h_ind_down'][k] = pre_ct_int[1]
  ret['w_ind_down'][k] = pre_ct_int[0]

  hm_tmp = np.zeros((int(self.opt.output_h / 2), int(self.opt.output_w / 2)),dtype=np.float32)
  draw_umich_gaussian_1d(hm_tmp, pre_ct_int, radius)
  ret['h_hm_down'][cls_id][:] = hm_tmp[:, pre_ct_int[0]]
  ret['w_hm_down'][cls_id][:] = hm_tmp[pre_ct_int[1], :]
  ret['h_hm_down'][cls_id][pre_ct_int[1]] = 0
  ret['w_hm_down'][cls_id][pre_ct_int[0]] = 0
