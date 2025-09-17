import json
import numpy as np
from xmodaler.evaluation.nms import multiclass_nms,remove_negtive
from xmodaler.evaluation.eval_map import eval_map
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
import os

giveup_score=0.5
giveup_nms=0.5
positive_iou=0.5
max_num=3

annfile='./open_source_dataset/CrowdMultiCaptions_MRC/JMDC4/mrc_test.json'
cap_file='./work_dirs/jmdc/Epoch.json'
det_file=cap_file.split('_cap')[0]+'_bbox.json'

with open(cap_file, 'r') as f:
    cap_result = json.load(f)
with open(det_file, 'r') as f:
    det_result = json.load(f)
coco_gt = COCO(annfile)
bbox_results={}
cap_results={}

for sub in det_result:
    bbox_results[sub['image_id']]={'scores':np.array(sub['scores']),'boxes':np.array(sub['boxes']),'labels':np.array(sub['labels'])}
for sub in cap_result:
    cap_results[sub['image_id']] = sub['caption']

label_process=str(max_num)+'_' + str(giveup_score)+'_'+str(giveup_nms)+'_'+str(positive_iou)
#########################   BBox NMS   #########################
rets_b, rets_c = remove_negtive(bbox_results,cap_results)
rets_b, rets_c = multiclass_nms(rets_b,rets_c, score_thresh=giveup_score, nms_thresh=giveup_nms, max_num=max_num)
#########################   BBox Score   #########################
score,eval_results,best_region = eval_map(rets_b,coco_gt,iou_thr=positive_iou)
mAP=score
pre=eval_results[0]['precision'][-1]
recall=eval_results[0]['recall'][-1]
Bbox_score={'mAP':mAP,'precision':pre,'recall':recall}
#########################   Caption Score   #########################
cocoEval = COCOEvalCap(coco_gt, cap_result, best_region)
cocoEval.evaluate()
ans=cocoEval.eval
print(Bbox_score)

#########################   save   #########################
bbox_save = cap_file.split('origin')[0] + label_process + '_bbox.json'
cap_save = cap_file.split('origin')[0] + label_process + '_cap.json'
save_b=[]
for sub in rets_b:
    aaa = {}
    aaa["image_id"] = sub
    aaa["boxes"] = rets_b[sub]['boxes'].tolist()
    aaa["labels"] = rets_b[sub]['labels'].tolist()
    aaa["scores"] = rets_b[sub]['scores'].tolist()
    save_b.append(aaa)
save_c=[]
for sub in rets_c:
    aaa={}
    aaa["image_id"] = sub
    aaa["caption"] = rets_c[sub]
    save_c.append(aaa)
json.dump(save_c, open(cap_save, "w"))
json.dump(save_b, open(bbox_save, "w"))
