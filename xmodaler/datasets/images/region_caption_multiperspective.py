# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li, Jianjie Luo
@contact: yehaoli.sysu@gmail.com, jianjieluo.sysu@gmail.com
"""
import os
import copy
import pickle
import random
import cv2
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, read_nz, dict_as_tensor, boxes_to_locfeats
from ..build import DATASETS_REGISTRY
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import str_to_pil_interp

__all__ = ["MRC_CrowdCaption_MP","MRC_CrowdscenesByTxtDataset_MP"]

@DATASETS_REGISTRY.register()
class MRC_CrowdCaption_MP:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        seq_per_img: int,
        reg_per_img: int,
        max_feat_num: int,
        max_seq_len: int,
        max_bbox_len: int,
        id2img_file: str,
        feats_folder: str,
        hrnet_folder: str,
        relation_file: str,
        gv_feat_file: str,
        attribute_file: str,
        input_size: int,
        mp_number: int,
    ):
        self.stage = stage
        self.anno_file = anno_file
        self.seq_per_img = seq_per_img
        self.reg_per_img = reg_per_img
        self.max_feat_num = max_feat_num
        self.mp_number = mp_number
        self.feats_folder = feats_folder
        self.hrnet_folder = hrnet_folder
        self.max_seq_len = max_seq_len
        self.max_bbox_len = max_bbox_len
        self.relation_file = relation_file
        self.gv_feat_file = gv_feat_file
        self.attribute_file = attribute_file
        self.input_size = input_size
        # 构建图像预处理单元
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=str_to_pil_interp('bicubic')),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
        )
        with open(id2img_file, 'r') as f:
            self.ids2path = json.load(f)

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER_TRAIN),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER_VAL),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER_TEST)
        }
        id2img_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER_TRAIN_ID),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER_VAL_ID),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER_TEST_ID)
        }
        ret = {
            "stage": stage,
            "id2img_file": id2img_files[stage],
            "anno_file": ann_files[stage],
            "seq_per_img": cfg.DATALOADER.SEQ_PER_SAMPLE,
            "reg_per_img": cfg.DATALOADER.REG_PER_SAMPLE,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "feats_folder": cfg.DATALOADER.FEATS_FOLDER,
            "hrnet_folder": cfg.DATALOADER.HRNET_FOLDER,
            "relation_file": cfg.DATALOADER.RELATION_FILE,
            "gv_feat_file": cfg.DATALOADER.GV_FEAT_FILE,
            "attribute_file": cfg.DATALOADER.ATTRIBUTE_FILE,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            "max_bbox_len": cfg.MODEL.MAX_BBOX_LEN,
            "input_size": cfg.DATALOADER.INPUT_SIZE,
            "mp_number": cfg.MODEL.MP_COUNT.NUM,
        }
        return ret

    def _preprocess_datalist(self, datalist):
        return datalist

    def load_data(self, cfg):
        def _load_pkl_file(filepath):
            return pickle.load(open(filepath, 'rb'), encoding='bytes') if len(filepath) > 0 else None

        datalist = pickle.load(open(self.anno_file, 'rb'), encoding='bytes')
        datalist = self._preprocess_datalist(datalist)
        ext_data = {
            "relation": _load_pkl_file(self.relation_file),
            "attribute": _load_pkl_file(self.attribute_file),
            "gv_feat": _load_pkl_file(self.gv_feat_file)
        }
        for i in range(len(datalist)):
            image_id = int(datalist[i]['image_id'])
            for data_type in ext_data:
                if ext_data[data_type] is not None:
                    if str(image_id) in ext_data[data_type]:
                        datalist[i][data_type] = ext_data[data_type][str(image_id)]
                    elif image_id in ext_data[data_type]:
                        datalist[i][data_type] = ext_data[data_type][image_id]
        '''
        if len(self.relation_file) > 0:
            relation = pickle.load(open(self.relation_file, 'rb'), encoding='bytes')
            for i in range(len(datalist)):
                image_id = int(datalist[i]['image_id'])
                if image_id in relation:
                    datalist[i]['relation'] = relation[image_id]
        '''
        # save_p=['00000827.jpg','00002968.jpg','00000164.jpg']
        # new_list=[datalist[0]]
        # for sub in datalist:
        #     if self.ids2path[sub['image_id']].split('/')[-1] not in save_p:
        #         continue
        #     new_list.append(sub)

        return datalist#[:100]
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        img_meta={}
        image_id = dataset_dict['image_id']

        if len(self.feats_folder) > 0:
            feat_path = os.path.join(self.feats_folder, image_id + '.npy')
            content = read_np(feat_path)
            att_feats = content['x'][0:self.max_feat_num].astype('float32')
            ret = { kfg.IDS: image_id, kfg.ATT_FEATS: att_feats }
        else:
            # 读取图像，并进行预处理
            image_path = self.ids2path[image_id]
            img = cv2.imread(image_path)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_input = self.transform(img)  # [3, 384, 384]，图像
            ret = {
                kfg.IDS: image_id,
                kfg.IMG_INPUT: img_input,
            }
        img_meta= {'image_id': dataset_dict['image_id'], 'img_h': dataset_dict['image_H'],'img_w': dataset_dict['image_W']}

        if len(self.hrnet_folder) > 0:
            feat_path = os.path.join(self.hrnet_folder, image_id + '.npz')
            hrnet_content = read_nz(feat_path)['feat']
            ret.update({kfg.PMP_FEATS: hrnet_content})

        if self.stage != 'train':
            g_tokens_cap_type = np.ones((self.max_seq_len), dtype=np.int64)
            ret.update({ kfg.G_TOKENS_CAP_TYPE: g_tokens_cap_type })
            tokens_bbox_ids = dataset_dict['tokens_loc_ids'].astype(np.int64)
            ret.update({'G_TOKENS_BBOX_IDS_ANS': tokens_bbox_ids})
            dict_as_tensor(ret)
            ret.update({kfg.IMG_META: img_meta})
            ids = image_id
            ret.update({kfg.IDS: ids})
            return ret

        if dataset_dict['count']>self.reg_per_img:
            dataset_dict['count'] = self.reg_per_img

        bbox_selects = list(range(len(dataset_dict['region_id'])))
        b_ids=[]
        for i in bbox_selects:
            b_ids.append(dataset_dict['region_id'][i])

        sent_num=len(dataset_dict['tokens_cap_ids'])
        if sent_num >= self.seq_per_img:
            sent_selects=random.sample(range(sent_num), self.seq_per_img)
        else:
            selects_origin = list(range(sent_num))
            sent_selects = list(range(sent_num))
            for i in range(self.seq_per_img - sent_num):
                num = random.choice(selects_origin)
                selects_origin.remove(num)
                sent_selects.append(num)
                if len(selects_origin)<1:
                    selects_origin = list(range(sent_num))

        tokens_cap_ids = np.stack([dataset_dict['tokens_cap_ids'][i, :].astype(np.int64) for i in sent_selects])
        target_cap_ids = np.stack([dataset_dict['target_cap_ids'][i, :].astype(np.int64) for i in sent_selects])
        g_tokens_cap_type = np.stack([np.ones((len(dataset_dict['tokens_cap_ids'][i, :]),), dtype=np.int64) for i in sent_selects])

        tokens_bbox_ids = dataset_dict['tokens_loc_ids'].astype(np.int64)
        target_bbox_ids = dataset_dict['target_loc_ids'].astype(np.int64)
        g_tokens_bbox_type = np.ones(len(dataset_dict['tokens_loc_ids']), dtype=np.int64)

        # target_pmp_ids = [ dataset_dict['pmp_target'][i].astype(np.int64) for i in bbox_selects ]
        target_count_ids = [np.array([dataset_dict['count']]).astype(np.int64)]

        ret.update({
            kfg.B_IDS: b_ids,
            kfg.IMG_H: dataset_dict['image_H'],
            kfg.IMG_W: dataset_dict['image_W'],
            kfg.SEQ_PER_SAMPLE: self.seq_per_img,
            kfg.REG_PER_SAMPLE: self.reg_per_img,
            kfg.G_TOKENS_CAP_IDS: tokens_cap_ids,
            kfg.G_TARGET_CAP_IDS: target_cap_ids,
            # kfg.G_TARGET_PMP: target_pmp_ids,
            kfg.G_TOKENS_CAP_TYPE: g_tokens_cap_type,
            kfg.G_TOKENS_BBOX_IDS: tokens_bbox_ids,
            kfg.G_TARGET_BBOX_IDS: target_bbox_ids,
            kfg.G_TOKENS_BBOX_TYPE: g_tokens_bbox_type,
            kfg.G_TARGET_COUNT_IDS: target_count_ids
        })
        dict_as_tensor(ret)
        ret.update({
            kfg.IMG_META: img_meta
        })
        return ret


@DATASETS_REGISTRY.register()
class MRC_CrowdscenesByTxtDataset_MP(MRC_CrowdCaption_MP):
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        seq_per_img: int,
        max_feat_num: int,
        max_seq_len: int,
        feats_folder: str,
        relation_file: str,
        gv_feat_file: str,
        attribute_file: str
    ):
        super(MRC_CrowdscenesByTxtDataset_MP, self).__init__(
            stage,
            anno_file,
            seq_per_img, 
            max_feat_num,
            max_seq_len,
            feats_folder,
            relation_file,
            gv_feat_file,
            attribute_file
        )
        assert self.seq_per_img == 1

    def _preprocess_datalist(self, datalist):
        if self.stage == 'train':
            expand_datalist = []
            for data in tqdm(datalist, desc='Expand Region Crowdscenes Dataset'):
                for token_id, target_id in zip(data['tokens_ids'], data['target_ids']):
                    expand_datalist.append({
                        'image_id': data['image_id'],
                        'tokens_ids': np.expand_dims(token_id, axis=0),
                        'target_ids': np.expand_dims(target_id, axis=0)
                    })
            return expand_datalist
        else:
            return datalist