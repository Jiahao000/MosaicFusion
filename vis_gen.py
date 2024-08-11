import argparse
import json
import os
import random

import cv2
import numpy as np
import pycocotools.mask as maskUtils
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from tqdm import tqdm

from mosaicfusion.utils import save_image
from mosaicfusion.vis import LVISVis


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize images and masks')
    
    parser.add_argument(
        '--ann_path',
        type=str,
        default='./output',
        help='input json annotation file path')
    
    parser.add_argument(
        '--img_path',
        type=str,
        default='./output',
        help='input image directory path')
    
    parser.add_argument(
        '--save_path',
        type=str,
        default='./output',
        help='output directory path')
    
    parser.add_argument(
        '-b',
        '--show_boxes',
        action='store_true',
        help='set True to show boxes')
    
    parser.add_argument(
        '-s',
        '--show_segms',
        action='store_true',
        help='set True to show segms')
    
    parser.add_argument(
        '-c',
        '--show_classes',
        action='store_true',
        help='set True to show classes')
    
    parser.add_argument(
        '-m',
        '--show_bimasks',
        action='store_true',
        help='set True to show binary masks')
    
    parser.add_argument(
        '--num_images',
        type=int,
        default=8,
        help='number of images to plot')
    
    parser.add_argument(
        '--img_id',
        type=int,
        nargs='+',
        default=None,
        help='specified image id(s) to plot (sample randomly if None)')
    
    parser.add_argument(
        '-g',
        '--save_grid',
        action='store_true',
        help='set True to save images as a grid')
    
    parser.add_argument(
        '-r',
        '--nrow',
        type=int,
        default=8,
        help='number of images displayed in each row of the grid')
    
    parser.add_argument(
        '--use_hw',
        action='store_true',
        help='set True to obtain images with a certain shape')
    
    parser.add_argument(
        '--height',
        type=int,
        default=768,
        help='image height')

    parser.add_argument(
        '--width',
        type=int,
        default=1024,
        help='image width')
    
    args = parser.parse_args()
    return args


def main():
    # init
    args = parse_args()
    vis = LVISVis(lvis_gt=args.ann_path, img_dir=args.img_path)
    if args.img_id is not None:
        if not isinstance(args.img_id, list):
            img_ids = [args.img_id]
        else:
            img_ids = args.img_id
    else:
        assert args.num_images > 0
        with open(args.ann_path, 'r') as f:
            data = json.load(f)
        img_ids_pool = list(range(len(data['images'])))
        img_ids_pool = [id + 1000000 for id in img_ids_pool]
        img_ids = np.random.choice(img_ids_pool, args.num_images, replace=False)
    if args.save_grid:
        grid_list = []
    img_filtered_ids = []
    for img_id in tqdm(img_ids, desc='Processing'):
        bimasks, fig, ax = vis.vis_img(
            img_id=img_id, 
            show_boxes=args.show_boxes,
            show_segms=args.show_segms,
            show_classes=args.show_classes)
        if args.show_bimasks:
            for i, mask in enumerate(bimasks):
                save_image(
                    mask,
                    save_path=os.path.join(args.save_path, f'img_{img_id}_mask_{i}.jpg'))
        img_name = str(img_id).zfill(12) + '.jpg'  # image names are 12 characters long
        load_image_path = os.path.join(args.img_path, img_name)
        image = Image.open(load_image_path).convert('RGB')
        if args.use_hw:
            if image.size[0] != args.width or image.size[1] != args.height:
                continue
        image.save(os.path.join(args.save_path, f'img_{img_id}.jpg'))
        save_blend_path = os.path.join(args.save_path, f'img_{img_id}_blend.jpg')
        fig.savefig(save_blend_path, bbox_inches='tight', dpi=300, pad_inches=0.0)
        blend = Image.open(save_blend_path).convert('RGB').resize(image.size)
        blend.save(save_blend_path)
        if args.save_grid:
            image = ToTensor()(image)
            blend = ToTensor()(blend)
            grid_list.append(image)
            grid_list.append(blend)
        img_filtered_ids.append(img_id)
    if args.save_grid:
        grid = torch.stack(grid_list, 0)
        grid = make_grid(grid, nrow=args.nrow)
        # to image
        grid = 255. * torch.einsum('chw->hwc', grid).numpy()
        img = Image.fromarray(grid.astype(np.uint8))
        save_grid_path = os.path.join(args.save_path, 'grid')
        os.makedirs(save_grid_path, exist_ok=True)
        img.save(os.path.join(save_grid_path, f'grid.png'))
    print('image_ids:', img_filtered_ids)
    print('Done!')


if __name__ == '__main__':
    main()
    