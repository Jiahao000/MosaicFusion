import argparse
import copy
import json
import math
import os
import random

import cv2
import numpy as np
import pycocotools.mask as maskUtils
from PIL import Image
from tqdm import tqdm

from mosaicfusion.utils import mosaic_coord, save_image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert generated images and masks to the json format')

    parser.add_argument(
        '--input_dir',
        type=str,
        default='./output',
        help='input directory to load images and masks')

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='output directory to save images and annotations')

    parser.add_argument(
        '--image_folder',
        type=str,
        default='train',
        help='folder to save images')

    parser.add_argument(
        '--filter_bg',
        action='store_true',
        help='set True to remove background categories same as foreground categories')

    parser.add_argument(
        '--use_bg',
        action='store_true',
        help='set True to use background categories in prompt')

    parser.add_argument(
        '--fg_category_path',
        type=str,
        default='data/lvis/meta/lvis_v1_train.json',
        help='foreground categories (lvis as default)')

    parser.add_argument(
        '--fg_main_category_type',
        type=str,
        nargs='+',
        default=['r'],
        choices=['f', 'c', 'r'],
        help='main foreground category type (lvis rare class as default)')
    
    parser.add_argument(
        '--fg_mixed_category_type',
        type=str,
        nargs='+',
        default=['f', 'c', 'r'],
        choices=['f', 'c', 'r'],
        help='mixed foreground category type (lvis rare, common, and frequent classes as default)')

    parser.add_argument(
        '--bg_category_path',
        type=str,
        default='data/places365/meta/categories_places365.txt',
        help='background categories (places365 as default)')

    parser.add_argument(
        '--bg_category_num',
        type=int,
        default=365,
        help='number of background categories (randomly sample a subset if < 365)')

    parser.add_argument(
        '--num_images',
        type=int,
        default=25,
        help='number of images to generate per category')
    
    parser.add_argument(
        '--num_images_per_split',
        type=int,
        default=25,
        help='number of images per split')
    
    parser.add_argument(
        '--height',
        type=int,
        default=384,
        help='image region height')

    parser.add_argument(
        '--width',
        type=int,
        default=512,
        help='image region width')
    
    parser.add_argument(
        '--num_regions',
        type=int,
        nargs='+',
        default=[4],
        choices=[1, 2, 4],
        help='number of regions to split per generated image')
    
    parser.add_argument(
        '--center_ratio_range',
        type=float,
        nargs=2,
        default=[0.75, 1.25],
        help='center ratio range of mosaic output (e.g., --center_ratio_range 0.75 1.25)')
    
    parser.add_argument(
        '--overlap',
        type=int,
        nargs=2,
        default=[48, 64],
        help='overlapped pixels (height, width) (e.g., --overlap 48 64)')

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='the seed (for reproducible sampling)')

    parser.add_argument(
        '-c',
        '--connectivity',
        type=int,
        default=4,
        help='connectivity for connected component analysis')

    parser.add_argument(
        '--height_range',
        type=int,
        nargs=2,
        default=[1, 768],
        help='keep masks within height range')

    parser.add_argument(
        '--width_range',
        type=int,
        nargs=2,
        default=[1, 1024],
        help='keep masks within width range')

    parser.add_argument(
        '--area_range',
        type=int,
        nargs=2,
        default=[9830, 186778],
        help='keep masks within area range (0.05-0.95 of the region area as default)')

    parser.add_argument(
        '-m',
        '--multi_masks',
        action='store_true',
        help='set True to allow multiple masks per image region')

    args = parser.parse_args()
    return args


def main():
    # init
    args = parse_args()
    num_splits = math.ceil(args.num_images / args.num_images_per_split)
    random.seed(args.seed)
    seeds = list(range(args.num_images_per_split))
    # load lvis cls info
    with open(args.fg_category_path, 'r') as f1:
        fg_data = json.load(f1)
    # main foreground categories
    fg_raw_list = []
    fg_list = []
    for c in fg_data['categories']:
        if c['frequency'] in args.fg_main_category_type:
            fg_raw_list.append(c['name'])
            fg_list.append(c)
    print(f'loaded {len(fg_list)} main foreground categories in total')
    # mixed foreground categories to sample
    fg_mixed_raw_list = []
    fg_mixed_list = []
    for c in fg_data['categories']:
        if c['frequency'] in args.fg_mixed_category_type:
            fg_mixed_raw_list.append(c['name'])
            fg_mixed_list.append(c)
    print(f'loaded {len(fg_mixed_list)} mixed foreground categories in total')
    if args.use_bg:
        # load places cls info
        with open(args.bg_category_path, 'r') as f2:
            bg_data = f2.readlines()
        bg_fns, bg_labels = zip(*[l.strip().split() for l in bg_data])
        if args.filter_bg:
            # filter fg in bg
            bg_filtered_fns = []
            for fn in bg_fns:
                filter_flag = False
                for fg_raw in fg_raw_list:
                    if fg_raw == fn.split('/')[2]:
                        print(f'filtering: {fg_raw} is in {fn}')
                        filter_flag = True
                        break
                if not filter_flag:
                    bg_filtered_fns.append(fn)
            print(f'filtered {len(bg_filtered_fns)} background from original {len(bg_fns)} background')
        else:
            bg_filtered_fns = bg_fns
        assert args.bg_category_num <= len(bg_filtered_fns)
        if args.bg_category_num < len(bg_filtered_fns):
            # randomly sample a subset
            bg_sample_fns = random.sample(bg_filtered_fns, args.bg_category_num)
        else:
            # use all background categories
            bg_sample_fns = bg_filtered_fns
        bg_raw_list = [fn.split('/', 2)[-1] for fn in bg_sample_fns]
        print(f'loaded {len(bg_raw_list)} from {len(bg_filtered_fns)} background categories in total')
    else:
        bg_raw_list = ['']
    img_list = []
    ann_list = []
    img_id = 1000000  # the max/min image_id in lvis is 581929/30, start from 1000000 to save generated images
    ann_id = 1
    # count the number of skipped images
    count1 = 0
    count2 = 0
    cat_list = []
    for cat in fg_data['categories']:
        cat['image_count'] = 0
        cat['instance_count'] = 0
        cat_list.append(cat)
    for split in tqdm(range(num_splits), desc='Processing split'):
        random.seed(args.seed)  # reset seed per split
        for fg in tqdm(fg_list, desc='Processing'): 
            for bg in bg_raw_list:
                # offset randomness
                seed_offset = split * args.num_images_per_split
                for _ in range(seed_offset):
                    if isinstance(args.num_regions, list):
                        num_regions = random.choice(args.num_regions)
                    else:
                        num_regions = args.num_regions
                    fgs = [fg]
                    fgs_mixed_ind = [random.randint(0, len(fg_mixed_list) - 1) for _ in range(num_regions - 1)]
                    fgs_mixed = [fg_mixed_list[ind] for ind in fgs_mixed_ind]
                    fgs.extend(fgs_mixed)
                    random.shuffle(fgs)
                    if num_regions == 1:
                        option = 0
                        canvas_height = args.height
                        canvas_width = args.width
                    elif num_regions == 2:
                        option = random.randint(0, 1)
                        if option == 0:
                            canvas_height = args.height
                            canvas_width = args.width * 2
                        else:
                            canvas_height = args.height * 2
                            canvas_width = args.width
                    elif num_regions == 4:
                        option = 0
                        canvas_height = args.height * 2
                        canvas_width = args.width * 2
                    else:
                        raise NotImplementedError
                    coords, _, _ = mosaic_coord(
                        center_ratio_range=tuple(args.center_ratio_range), 
                        img_shape_hw=(args.height, args.width), 
                        overlap_hw=tuple(args.overlap), 
                        num_regions=num_regions,
                        region_option=option) 
                for s in seeds:
                    if isinstance(args.num_regions, list):
                        num_regions = random.choice(args.num_regions)
                    else:
                        num_regions = args.num_regions
                    fgs = [fg]
                    fgs_mixed_ind = [random.randint(0, len(fg_mixed_list) - 1) for _ in range(num_regions - 1)]
                    fgs_mixed = [fg_mixed_list[ind] for ind in fgs_mixed_ind]                      
                    fgs.extend(fgs_mixed)
                    random.shuffle(fgs)
                    fgs_raw = [fgd['name'] for fgd in fgs]
                    load_path = os.path.join(args.input_dir, fg['name'], bg, str(s + seed_offset), '-'.join(fgs_raw))
                    if num_regions == 1:
                        option = 0
                        canvas_height = args.height
                        canvas_width = args.width
                    elif num_regions == 2:
                        option = random.randint(0, 1)
                        if option == 0:
                            canvas_height = args.height
                            canvas_width = args.width * 2
                        else:
                            canvas_height = args.height * 2
                            canvas_width = args.width
                    elif num_regions == 4:
                        option = 0
                        canvas_height = args.height * 2
                        canvas_width = args.width * 2
                    else:
                        raise NotImplementedError
                    coords, _, _ = mosaic_coord(
                        center_ratio_range=tuple(args.center_ratio_range),
                        img_shape_hw=(args.height, args.width),
                        overlap_hw=tuple(args.overlap),
                        num_regions=num_regions,
                        region_option=option)
                    if s + seed_offset < args.num_images:
                        load_image_path = os.path.join(load_path, 'image.jpg')
                        image = np.array(Image.open(load_image_path).convert('RGB'))
                        filtered_masks_list = []
                        for region_idx in range(num_regions):
                            load_mask_path = os.path.join(load_path, f'region{region_idx}_mask.jpg')
                            region_mask = np.array(Image.open(load_mask_path).convert('1'), dtype=np.uint8) * 255
                            mask = np.full((canvas_height, canvas_width), 0, dtype=region_mask.dtype)
                            mask[coords[region_idx][0]:coords[region_idx][1], coords[region_idx][2]:coords[region_idx][3]] = region_mask
                            output = cv2.connectedComponentsWithStats(mask, args.connectivity, cv2.CV_32S)
                            (numLabels, labels, stats, centroids) = output
                            filtered_masks = []
                            for l in range(1, numLabels):  # 0 is background
                                x = stats[l, cv2.CC_STAT_LEFT]
                                y = stats[l, cv2.CC_STAT_TOP]
                                w = stats[l, cv2.CC_STAT_WIDTH]
                                h = stats[l, cv2.CC_STAT_HEIGHT]
                                area = stats[l, cv2.CC_STAT_AREA]
                                # filter criterion
                                keepWidth = w >= args.width_range[0] and w <= args.width_range[1]
                                keepHeight = h >= args.height_range[0] and h <= args.height_range[1]
                                keepArea = area >= args.area_range[0] and area <= args.area_range[1]
                                if all((keepWidth, keepHeight, keepArea)):
                                    filtered_mask = (labels == l).astype(np.uint8)
                                    filtered_masks.append(filtered_mask)
                            if len(filtered_masks) == 1 or (len(filtered_masks) > 1 and args.multi_masks):
                                filtered_masks_list.append(filtered_masks)
                            elif len(filtered_masks) > 1:
                                count1 += 1
                                print(f'skip the image with multiple masks in region {region_idx}: {load_image_path}')
                                break
                            else:
                                assert len(filtered_masks) == 0
                                count2 += 1
                                print(f'skip the image without any masks in region {region_idx}: {load_image_path}')
                                break
                        if len(filtered_masks_list) == num_regions:
                            for region_idx in range(num_regions):
                                for filtered_mask in filtered_masks_list[region_idx]:
                                    filtered_mask = np.asfortranarray(filtered_mask)
                                    rle = maskUtils.encode(filtered_mask)
                                    rle['counts'] = rle['counts'].decode()
                                    area = maskUtils.area(rle)
                                    bbox = maskUtils.toBbox(rle)
                                    cur_ann_dict = {'id': ann_id,
                                                    'image_id': img_id,
                                                    'category_id': fgs[region_idx]['id'],
                                                    'segmentation': rle,
                                                    'area': float(area),
                                                    'bbox': bbox.tolist()}
                                    ann_id += 1
                                    assert cat_list[fgs[region_idx]['id'] - 1]['id'] == fgs[region_idx]['id']
                                    cat_list[fgs[region_idx]['id'] - 1]['instance_count'] += 1
                                    ann_list.append(cur_ann_dict)
                            img_name = str(img_id).zfill(12) + '.jpg'  # image names are 12 characters long
                            cur_img_dict = {'id': img_id,
                                            'height': image.shape[0],
                                            'width': image.shape[1],
                                            'coco_url': args.image_folder + '/' + img_name,
                                            'neg_category_ids': [],
                                            'not_exhaustive_category_ids': []}
                            img_id += 1
                            fgs_cat_id = list(set([fgd['id'] for fgd in fgs]))
                            for cat_id in fgs_cat_id:
                                assert cat_list[cat_id - 1]['id'] == cat_id
                                cat_list[cat_id - 1]['image_count'] += 1
                            img_list.append(cur_img_dict)
                            # save jpg images
                            save_image_path = os.path.join(args.output_dir, args.image_folder)
                            os.makedirs(save_image_path, exist_ok=True)
                            save_image(image, save_path=os.path.join(save_image_path, img_name))
    # save json annotations
    ann_dict = dict()
    ann_dict['images'] = img_list
    ann_dict['annotations'] = ann_list
    cat_list = [cat for cat in cat_list if cat['image_count'] != 0]
    ann_dict['categories'] = cat_list
    save_ann_path = os.path.join(args.output_dir, 'annotations')
    os.makedirs(save_ann_path, exist_ok=True)
    with open(os.path.join(save_ann_path, 'lvis_v1_train_mosaicfusion.json'), 'w') as f:
        json.dump(ann_dict, f)
    print(f'skipped {count1} images with multiple masks, {count2} images without any masks')
    print(f'Done: {len(img_list)} images, {len(ann_list)} annotations, {len(cat_list)} categories')


if __name__ == '__main__':
    main()
