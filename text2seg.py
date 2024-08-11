import argparse
import json
import math
import os
import random

import nltk
nltk.download('wordnet')
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, LMSDiscreteScheduler, DPMSolverMultistepScheduler
from nltk.corpus import wordnet as wn
from tqdm import tqdm

from mosaicfusion.bilateral_solver import bilateral_solver_output
from mosaicfusion.canvas import MosaicFusionPipeline, MosaicFusionXLPipeline, \
    Text2ImageRegion, Text2ImageRegionXL
from mosaicfusion.utils import AttentionStore, get_cross_attention_mask, mosaic_coord, \
    register_attention_control, save_image, text2image_ldm_stable


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate images and masks with MosaicFusion')

    parser.add_argument(
        '--prompt',
        type=str,
        nargs='+',
        default='',
        help='the prompt template to render')

    parser.add_argument(
        '--fg_def',
        action='store_true',
        help='set True to append the foreground definition after the foreground category')

    parser.add_argument(
        '--bg_def',
        action='store_true',
        help='set True to append the background definition after the background category')

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
        '--model_id',
        type=str,
        default='CompVis/stable-diffusion-v1-4',
        help='model id for diffusion model')
    
    parser.add_argument(
        '--low_resource',
        action='store_true',
        help='set True for running on 12GB GPU')
    
    parser.add_argument(
        '--run_standard_sd',
        action='store_true',
        help='set True to run the standard Stable Diffusion model without mixture')

    parser.add_argument(
        '--diffusion_steps',
        type=int,
        default=50,
        help='number of diffusion steps')

    parser.add_argument(
        '--scale',
        type=float,
        default=7.5,
        help='unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))')

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='the seed (for reproducible sampling)')

    parser.add_argument(
        '--seed_offset',
        type=int,
        default=0,
        help='start from what seed to generate images (for reproducible sampling)')
    
    parser.add_argument(
        '--res_idx',
        type=int,
        nargs='+',
        default=[1, 2, 3],
        choices=[0, 1, 2, 3],
        help='resolution index of cross-attention maps (0: 1/64, 1: 1/32, 2: 1/16, 3: 1/8)')

    parser.add_argument(
        '--from_where',
        type=str,
        nargs='+',
        default=['up', 'down'],
        choices=['up', 'mid', 'down'],
        help='from where to get cross-attention map')

    parser.add_argument(
        '--start_step',
        type=int,
        default=0,
        help='start step to get cross-attention map')

    parser.add_argument(
        '--end_step',
        type=int,
        default=50,
        help='end step to get cross-attention map')

    parser.add_argument(
        '--attn_threshold_type',
        type=str,
        default='otsu',
        choices=['simple', 'otsu'],
        help='threshold type to binarize the cross-attention map')

    parser.add_argument(
        '--attn_threshold',
        type=float,
        default=0.3,
        help='threshold to binarize the cross-attention map (only be called in simple mode)')

    parser.add_argument(
        '--bilateral_solver_times',
        type=int,
        default=2,
        help='how many times (>=0) to apply the bilateral solver')

    parser.add_argument(
        '--sigma_spatial',
        type=float,
        default=16,
        help='sigma spatial in the bilateral solver')

    parser.add_argument(
        '--sigma_luma',
        type=float,
        default=16,
        help='sigma luma in the bilateral solver')

    parser.add_argument(
        '--sigma_chroma',
        type=float,
        default=8,
        help='sigma chroma in the bilateral solver')

    parser.add_argument(
        '--bfs_threshold',
        type=float,
        default=0.5,
        help='threshold to binarize the bilateral solver map')

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='set True to save intermediate results')

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='output directory')

    args = parser.parse_args()
    return args


def main():
    # init
    args = parse_args()
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.run_standard_sd:
        if args.model_id.endswith('stable-diffusion-xl-base-1.0'):
            ldm_stable = StableDiffusionXLPipeline.from_pretrained(
                args.model_id, use_safetensors=True).to(device)
        else:
            ldm_stable = StableDiffusionPipeline.from_pretrained(
                args.model_id).to(device)
    else:
        # create scheduler and model (similar to StableDiffusionPipeline)
        if args.model_id.endswith('stable-diffusion-xl-base-1.0'):
            ldm_stable = MosaicFusionXLPipeline.from_pretrained(
                args.model_id, use_safetensors=True).to(device)
        else:
            ldm_stable = MosaicFusionPipeline.from_pretrained(
                args.model_id).to(device)
    
    tokenizer = ldm_stable.tokenizer
    random.seed(args.seed)
    seeds = list(range(args.num_images))
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
        bg_list = []
        for bg in bg_raw_list:
            bgs = bg.split('/')
            assert len(bgs) == 1 or len(bgs) == 2
            if args.bg_def:
                if len(wn.synsets(bgs[0], pos=wn.NOUN)):
                    bgs[0] = bgs[0] + ', ' + wn.synsets(bgs[0], pos=wn.NOUN)[0].definition()
            if len(bgs) == 1:
                bg_list.append(bgs[0].replace('_', ' '))
            else:
                bg_list.append((bgs[-1] + ' ' + bgs[0]).replace('_', ' '))
        print(f'loaded {len(bg_list)} from {len(bg_filtered_fns)} background categories in total')
    else:
        bg_raw_list = ['']
        bg_list = ['scene']
    # instantiate prompt
    for p in tqdm(args.prompt, desc='Processing'):
        print(f'\nprompt template: {p}')
        for i, fgd in enumerate(tqdm(fg_list, desc='Processing foreground')):       
            for j, bg in enumerate(bg_list):
                # offset randomness
                for _ in range(args.seed_offset):
                    if isinstance(args.num_regions, list):
                        num_regions = random.choice(args.num_regions)
                    else:
                        num_regions = args.num_regions
                    fgs = [fgd]
                    fgs_mixed_ind = [random.randint(0, len(fg_mixed_list) - 1) for _ in range(num_regions - 1)]
                    fgs_mixed = [fg_mixed_list[ind] for ind in fgs_mixed_ind]
                    fgs.extend(fgs_mixed)
                    random.shuffle(fgs)
                    if args.run_standard_sd:
                        assert num_regions == 1
                        canvas_height = args.height
                        canvas_width = args.width
                        coords, paste_coords, crop_coords = mosaic_coord(
                            center_ratio_range=tuple(args.center_ratio_range), 
                            img_shape_hw=(args.height, args.width), 
                            overlap_hw=tuple(args.overlap), 
                            num_regions=num_regions)
                    else:
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
                        coords, paste_coords, crop_coords = mosaic_coord(
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
                    fgs = [fgd]
                    fgs_mixed_ind = [random.randint(0, len(fg_mixed_list) - 1) for _ in range(num_regions - 1)]
                    fgs_mixed = [fg_mixed_list[ind] for ind in fgs_mixed_ind]
                    fgs.extend(fgs_mixed)
                    random.shuffle(fgs)
                    fgs_raw = [fg['name'] for fg in fgs]
                    fgps = [fg['name'].replace('_', ' ') for fg in fgs]
                    if args.fg_def:
                        fgps = [(fgp + ', ' + fg['def']) for fgp, fg in zip(fgps, fgs)]
                    prompts = [p.replace('category', fgp).replace('scene', bg) for fgp in fgps]
                    [print(f'\nprocessing prompt: {prompt}') for prompt in prompts]
                    prompt_tokens_list = [tokenizer.encode(prompt) for prompt in prompts]
                    fg_tokens_list = [tokenizer.encode(fg['name'].replace('_', ' ').replace(')', '')) for fg in fgs]
                    token_inds = [prompt_tokens.index(fg_tokens[-2]) for prompt_tokens, fg_tokens in zip(prompt_tokens_list, fg_tokens_list)]
                    [print(f'\ncross-attention map: word: {tokenizer.decode(int(fg_tokens[-2]))}, token index: {token_ind}') for fg_tokens, token_ind in zip(fg_tokens_list, token_inds)]
                    # save path: foreground main category/(background)/seed/foreground mixed categories
                    save_path = os.path.join(args.output_dir, fg_raw_list[i], bg_raw_list[j], str(s + args.seed_offset), '-'.join(fgs_raw))
                    os.makedirs(save_path, exist_ok=True)
                    controller = AttentionStore(
                        start_step=args.start_step,
                        end_step=args.end_step,
                        num_regions=num_regions,
                        low_resource=args.low_resource)
                    if args.run_standard_sd:
                        # standard generation with Stable Diffusion 
                        assert num_regions == 1
                        canvas_height = args.height
                        canvas_width = args.width
                        coords, paste_coords, crop_coords = mosaic_coord(
                            center_ratio_range=tuple(args.center_ratio_range), 
                            img_shape_hw=(args.height, args.width), 
                            overlap_hw=tuple(args.overlap), 
                            num_regions=num_regions)
                        g_cpu = torch.Generator().manual_seed(s + args.seed_offset)
                        image, x_t = text2image_ldm_stable(
                            ldm_stable, prompts, controller, height=args.height, width=args.width,
                            num_inference_steps=args.diffusion_steps, guidance_scale=args.scale,
                            generator=g_cpu, latent=None, low_resource=args.low_resource)
                    else:
                        # multi-object generation with MosaicFusion
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
                        coords, paste_coords, crop_coords = mosaic_coord(
                            center_ratio_range=tuple(args.center_ratio_range), 
                            img_shape_hw=(args.height, args.width), 
                            overlap_hw=tuple(args.overlap), 
                            num_regions=num_regions,
                            region_option=option)
                        if args.model_id.endswith('stable-diffusion-xl-base-1.0'):
                            region_cfgs = [
                                Text2ImageRegionXL(
                                    coord[0], coord[1], coord[2], coord[3], 
                                    guidance_scale=args.scale, prompt=prompt
                                    ) for coord, prompt in zip(coords, prompts)
                                ]
                        else:
                            region_cfgs = [
                                Text2ImageRegion(
                                    coord[0], coord[1], coord[2], coord[3], 
                                    guidance_scale=args.scale, prompt=prompt
                                    ) for coord, prompt in zip(coords, prompts)
                                ]
                        register_attention_control(ldm_stable, controller, num_regions)
                        image = ldm_stable(
                            controller=controller,
                            canvas_height=canvas_height,
                            canvas_width=canvas_width,
                            regions=region_cfgs,
                            num_inference_steps=args.diffusion_steps,
                            seed=(s + args.seed_offset),
                            low_resource=args.low_resource)["sample"]
                    final_masks = []
                    for region_idx in range(num_regions):
                        region_height = coords[region_idx][1] - coords[region_idx][0]
                        region_width = coords[region_idx][3] - coords[region_idx][2]
                        region_latent_res_list = []
                        for res_idx in range(4):
                            if res_idx == 0:
                                region_latent_height = region_height // 8
                                region_latent_width = region_width // 8
                            else:
                                region_latent_height = math.ceil(region_latent_height / 2)
                                region_latent_width = math.ceil(region_latent_width / 2)
                            region_latent_res_list.append((region_latent_height, region_latent_width))
                        region_latent_res_list = region_latent_res_list[::-1]
                        region_latent_res_list = [region_latent_res_list[res_idx] for res_idx in args.res_idx]
                        assert len(region_latent_res_list) == len(args.res_idx)
                        _, masks = get_cross_attention_mask(
                            prompts=[prompts[region_idx]], attention_store=controller, 
                            height=region_height, width=region_width, res=region_latent_res_list, 
                            from_where=args.from_where, threshold_type=args.attn_threshold_type, threshold=args.attn_threshold, 
                            save_path=save_path, verbose=args.verbose, token_ind=token_inds[region_idx], region_idx=region_idx)
                        assert len(masks) == 1
                        final_mask = masks[0]
                        # bilateral solver
                        if args.bilateral_solver_times:
                            region_image = image[0][coords[region_idx][0]:coords[region_idx][1], coords[region_idx][2]:coords[region_idx][3]]
                            for t in range(args.bilateral_solver_times):
                                output_solver, binary_solver = bilateral_solver_output(
                                    region_image, (255 * final_mask), sigma_spatial=args.sigma_spatial,
                                    sigma_luma=args.sigma_luma, sigma_chroma=args.sigma_chroma)
                                output = output_solver / output_solver.max()
                                final_mask = np.where(output > args.bfs_threshold, 1, 0)
                                blend_region_image = region_image * np.expand_dims(final_mask, -1).repeat(3, axis=-1)
                                if args.verbose:
                                    save_image(
                                        255 * final_mask,
                                        save_path=os.path.join(save_path, f'region{region_idx}_mask_avg_bfs{t+1}.jpg'))
                                    save_image(
                                        blend_region_image,
                                        save_path=os.path.join(save_path, f'region{region_idx}_blend_avg_bfs{t+1}.jpg'))
                        # save final region mask
                        save_image(255 * final_mask, save_path=os.path.join(save_path, f'region{region_idx}_mask.jpg'))
                        final_masks.append(final_mask)
                    # save final global image
                    save_image(image[0], save_path=os.path.join(save_path, 'image.jpg'))
                    # final global mask
                    final_global_mask = np.full((canvas_height, canvas_width), 0, dtype=final_masks[0].dtype)
                    for region_idx in range(num_regions):
                        final_global_mask[paste_coords[region_idx][0]:paste_coords[region_idx][1], paste_coords[region_idx][2]:paste_coords[region_idx][3]] = \
                            final_masks[region_idx][crop_coords[region_idx][0]:crop_coords[region_idx][1], crop_coords[region_idx][2]:crop_coords[region_idx][3]]
                    # save final global mask
                    save_image(255 * final_global_mask, save_path=os.path.join(save_path, f'mask.jpg'))
                    # save final global blended image
                    save_image(
                        image[0] * np.expand_dims(final_global_mask, -1).repeat(3, axis=-1),
                        save_path=os.path.join(save_path, 'blend.jpg'))


if __name__ == '__main__':
    main()
