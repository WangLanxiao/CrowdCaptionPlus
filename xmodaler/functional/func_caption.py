# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import numpy as np
import torch
def decode_sequence(vocab, seq):
    N, R, T = seq.size()
    ANS=[]
    for n in range(N):
        sents = []
        for r in range(R):
            words = []
            for t in range(T):
                ix = seq[n, r, t]
                if ix == 0:
                    break
                words.append(vocab[ix])
            sent = ' '.join(words)
            sents.append(sent)
        ANS.append(sents)
    return ANS
def map_convert(x,y,w,h,W,H,Scale, mode='xywhimg2map'):
    if mode == 'xywhimg2map':
        x1_o = int(x/W*Scale)+1
        y1_o = int(y/H*Scale)+1
        x2_o = int((x+w-1)/W*Scale)+1
        y2_o = int((y+h-1)/H*Scale)+1
        bbox_output = [x1_o, y1_o, x2_o, y2_o]
    elif mode == 'xywhmap2img':
        x1_o = int(x / Scale * W)
        y1_o = int(y / Scale * H)
        x2_o = int((x + w - 1) / Scale * W)
        y2_o = int((y + h - 1) / Scale * H)
        bbox_output = [x1_o, y1_o, x2_o, y2_o]
    elif mode == 'xywhimg2map':
        x1_o = int(x / W * Scale) + 1
        y1_o = int(y / H * Scale) + 1
        x2_o = int(w / W * Scale) + 1
        y2_o = int(h / H * Scale) + 1
        bbox_output = [x1_o, y1_o, x2_o, y2_o]
    elif mode == 'xyxymap2img':
        x1_o = int(x / Scale * W)
        y1_o = int(y / Scale * H)
        x2_o = int(w / Scale * W)
        y2_o = int(h / Scale * H)
        bbox_output = [x1_o, y1_o, x2_o, y2_o]
    else:
        raise Exception("Unsupported mode")
    return bbox_output


def decode_bbox(vocab, seq, bbox_score):
    N, T = seq.size()
    sents = []
    for n in range(N):
        words=[]
        score = []
        for t in range(T):
            ix = seq[n, t]
            if ix == 0:
                break
            words.append(int(vocab[ix]))
            score.append(bbox_score[n,t])
        if len(words)%4!=0:
            words=words[:len(words)-len(words)%4]
            score=score[:len(score)-len(score)%4]
            print('exist some bbox not 4  times')
        words=np.array(words).reshape(-1, 4)
        score=np.array(score).reshape(-1, 4).sum(-1)
        bbox=np.concatenate((words, np.expand_dims(score, -1)), -1).tolist()
        sents.append(bbox)
    return sents

def bbox2result(bboxes, labels, num_classes):
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy().reshape(-1,5)
            labels = labels.detach().cpu().numpy()
        return [bboxes.reshape(-1,5)[labels == i, :] for i in range(num_classes)]

def _meshgrid(x, y, row_major=True):
    yy, xx = torch.meshgrid(y, x)
    if row_major:
        # warning .flatten() would cause error in ONNX exporting
        # have to use reshape here
        return xx.reshape(-1), yy.reshape(-1)

    else:
        return yy.reshape(-1), xx.reshape(-1)

def single_level_grid_priors(featmap_size,
                             dtype=torch.float32,
                             device='cuda',
                             offset=0.5,
                             with_stride=False):
    feat_h, feat_w = featmap_size
    stride_w, stride_h = [32,32]
    shift_x = (torch.arange(0, feat_w, device=device) +
               offset) * stride_w
    shift_x = shift_x.to(dtype)

    shift_y = (torch.arange(0, feat_h, device=device) +
               offset) * stride_h
    shift_y = shift_y.to(dtype)
    shift_xx, shift_yy = _meshgrid(shift_x, shift_y)
    if not with_stride:
        shifts = torch.stack([shift_xx, shift_yy], dim=-1)
    else:
        stride_w = shift_xx.new_full((shift_xx.shape[0],),
                                     stride_w).to(dtype)
        stride_h = shift_xx.new_full((shift_yy.shape[0],),
                                     stride_h).to(dtype)
        shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h],
                             dim=-1)
    all_points = shifts.to(device)
    return all_points

def filter_scores_and_topk(scores, score_thr, topk, results=None):
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)
    num_topk = min(topk, valid_idxs.size(0))
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)
    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                      f'but get {type(results)}.')
    return scores, labels, keep_idxs, filtered_results

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """

    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            # speed up
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            return bboxes
        # clip bboxes with dynamic `min` and `max` for onnx
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape],
                           dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)
    return bboxes

def bbox_post_process( mlvl_scores,
                       mlvl_labels,
                       mlvl_bboxes,
                       mlvl_score_factors,
                       scale_factor,
                       with_nms=True):

    mlvl_bboxes = torch.cat(mlvl_bboxes)
    mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
    mlvl_scores = torch.cat(mlvl_scores)
    mlvl_labels = torch.cat(mlvl_labels)

    # mlvl_score_factors = torch.cat(mlvl_score_factors)
    # mlvl_scores = mlvl_scores * mlvl_score_factors

    return mlvl_bboxes, mlvl_scores, mlvl_labels

def get_bboxes_single(cls_score_list, bbox_pred_list,score_factor_list, all_level_points,img_meta):
    img_shape = (img_meta['img_h'], img_meta['img_w'],3)
    nms_pre = 1
    cls_out_channels =1
    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_labels = []
    mlvl_score_factors = []
    for level_idx, (cls_score, bbox_pred, score_factor, priors) in enumerate(zip([cls_score_list], [bbox_pred_list], [score_factor_list], [all_level_points])):
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, cls_out_channels)
        scores = cls_score.sigmoid()
        results = filter_scores_and_topk(scores,0.05, nms_pre,dict(bbox_pred=bbox_pred, priors=priors))
        scores, labels, keep_idxs, filtered_results = results
        bbox_pred = filtered_results['bbox_pred']
        priors = filtered_results['priors']
        score_factor = score_factor[keep_idxs]
        bboxes = distance2bbox(priors, bbox_pred, max_shape=(384,384,3))
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)
        mlvl_labels.append(labels)
        mlvl_score_factors.append(score_factor)
    scale_factor=np.array([img_shape[0] / 384.0, img_shape[1] / 384.0, img_shape[0] / 384.0, img_shape[1] / 384.0])
    bbox_result=bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,mlvl_score_factors, scale_factor)
    return bbox_result

def decode_fcosbbox(img_metas,ids, bbox_results,cls_score,centerness):
    b,_, _, _ = cls_score.shape
    cls_score = cls_score.reshape(b , -1, 12, 12)
    bbox_results = bbox_results.reshape(b, -1, 12, 12)
    centerness = centerness.reshape(b , -1, 12, 12)

    featmap_sizes = cls_score.shape[-2:]
    all_level_points = single_level_grid_priors(
        featmap_sizes,
        dtype=cls_score.dtype,
        device=cls_score.device)
    result_list = []
    cls_score_list = list(cls_score)
    bbox_pred_list = list(bbox_results)
    score_factor_list = list(centerness)

    for idx in range(len(ids)):
        img_id = ids[idx]
        img_meta = img_metas[img_id]
        results = get_bboxes_single(cls_score_list[idx], bbox_pred_list[idx],
                                          score_factor_list[idx], all_level_points,
                                          img_meta)
        result_list.append(results)
    num_classes=1
    bbox_results=[]
    for sub in result_list:
        det_bboxes = torch.cat([sub[0].reshape(-1),sub[1]],dim=-1)
        det_labels = sub[2]
        bbox_results.append(bbox2result(det_bboxes, det_labels, num_classes))
    return bbox_results

def decode_mp_count(seq):
    N, T = seq.size()
    sents = []
    for n in range(N):
        sents.append(int(seq[n]))
    return sents

def decode_origin_bbox(seq):
    N, T = seq.size()
    sents = []
    for n in range(N):
        sents.append(int(seq[n]))
    return sents

def decode_sequence_bert(tokenizer, seq, sep_token_id):
    N, T = seq.size()
    seq = seq.data.cpu().numpy()
    sents = []
    for n in range(N):
        words = []
        for t in range(T):
            ix = seq[n, t]
            if ix == sep_token_id:
                break
            words.append(tokenizer.ids_to_tokens[ix])
        sent = tokenizer.convert_tokens_to_string(words)
        sents.append(sent)
    return sents