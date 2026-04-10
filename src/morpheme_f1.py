import re
import sys
sys.path.insert(1, '2022SegmentationST/evaluation')
from evaluate import n_correct
morpheme_boundary = re.compile(r'(?<=[.+-=])|\|')

def compute_f1(corr_segmentations: list[str], pred_segmentations: list[str]) -> dict[str, float]:
  total_corr = 0
  total_pred = 0
  total_overlaps = 0
  for pred_segmentation, corr_segmentation in zip(pred_segmentations, corr_segmentations, strict=True):
    corr_segments = morpheme_boundary.split(corr_segmentation.replace(' @@', '|'))
    pred_segments = morpheme_boundary.split(pred_segmentation.replace(' @@', '|'))
    total_corr += len(corr_segments)
    total_pred += len(pred_segments)
    n_overlaps = n_correct('|'.join(corr_segments), '|'.join(pred_segments))
    total_overlaps += n_overlaps
  precision = 100 * total_overlaps / total_pred
  recall = 100 * total_overlaps / total_corr
  if precision+recall == 0:
      f_measure = .0
  else:
      f_measure = 2 * precision * recall / (precision + recall)
  return {"precision": precision, "recall": recall, "f_measure": f_measure}
