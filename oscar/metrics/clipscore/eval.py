from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--testfile', type=str, default='../../eval_results/vinvl_result.json') 
parser.add_argument('--annofile', type=str, default='../../datasets/coco_caption/test_caption_coco_format.json' ) # or 'naive'
args = parser.parse_args()

results_file = args.testfile
annotation_file = args.annofile

# create coco object and coco_result object
coco = COCO(annotation_file)
coco_result = coco.loadRes(results_file)

# create coco_eval object by taking coco and coco_result
coco_eval = COCOEvalCap(coco, coco_result)

# evaluate on a subset of images by setting
# coco_eval.params['image_id'] = coco_result.getImgIds()
# please remove this line when evaluating the full validation set
coco_eval.params['image_id'] = coco_result.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
coco_eval.evaluate()

# print output evaluation scores
for metric, score in coco_eval.eval.items():
    print(f'{metric}: {score:.3f}')