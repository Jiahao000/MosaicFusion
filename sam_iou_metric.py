import argparse
import json
import os
import sys
# sys.path.append("..")

import cv2
import numpy as np
import pycocotools.mask as maskUtils
import torch
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate mean IoU using SAM as ground truth')
    
    parser.add_argument(
        '--ann_path',
        type=str,
        default='data',
        help='input json annotation file path')
    
    parser.add_argument(
        '--img_path',
        type=str,
        default='data',
        help='input image directory path')

    parser.add_argument(
        '--sam_ckpt',
        type=str,
        default='segment-anything/checkpoints/sam_vit_h_4b8939.pth',
        help='sam checkpoint')

    parser.add_argument(
        '--model_type',
        type=str,
        default='vit_h',
        help='sam model type')

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='device (cuda/cpu)')
    
    args = parser.parse_args()
    return args


def main():
    # init
    args = parse_args()
    # sam
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_ckpt)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)
    # data
    with open(args.ann_path, 'r') as f:
        data = json.load(f)
    iou_list = []
    for img_dict in tqdm(data['images'], desc='Processing'):
        image = cv2.imread(os.path.join(args.img_path, img_dict['coco_url']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        dt_masks = []
        input_boxes = []
        for ann_dict in data['annotations']:
            if ann_dict['image_id'] == img_dict['id']:
                dt_masks.append(ann_dict['segmentation'])
                input_box = [ann_dict['bbox'][0], ann_dict['bbox'][1], ann_dict['bbox'][0] + ann_dict['bbox'][2], ann_dict['bbox'][1] + ann_dict['bbox'][3]]
                input_boxes.append(input_box)
        input_boxes = torch.tensor(input_boxes, device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        gt_masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
        )
        # mask -> rle
        gt_masks = [np.asfortranarray(mask.squeeze().cpu().numpy()) for mask in gt_masks]
        gt_masks = [maskUtils.encode(mask) for mask in gt_masks]
        for mask in gt_masks:
            mask['counts'] = mask['counts'].decode()
        is_crowd = [0]
        # iou per image
        iou = [maskUtils.iou([dt], [gt], is_crowd) for dt, gt in zip(dt_masks, gt_masks)]
        iou_list.append(np.mean(iou))
    mean_iou = sum(iou_list) / len(iou_list)
    print('mIoU: ', mean_iou)
    print('Done!')
        

if __name__ == '__main__':
    main()
    