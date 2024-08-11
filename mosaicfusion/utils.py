import abc
import os
import random
from typing import Optional, Union, Tuple, List, Callable, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from PIL import Image
from tqdm.notebook import tqdm


def diffusion_step(model, controller, latents, context, t, guidance_scale, extra_step_kwargs, low_resource=False):
    if low_resource:
        latent_model_input = latents
        latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
        noise_pred_uncond = model.unet(latent_model_input, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latent_model_input, t, encoder_hidden_states=context[1])["sample"]
    else:
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
        noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
            dtype=model.unet.dtype
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
    eta: float = 0.0,
):
    register_attention_control(model, controller)
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # set timesteps
    model.scheduler.set_timesteps(num_inference_steps, device=model.device)
    # prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = model.prepare_extra_step_kwargs(generator, eta)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, extra_step_kwargs, low_resource)
    
    image = latent2image(model.vae, latents)
  
    return image, latent


class CrossAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, attnstore, place_in_unet):
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        region_idx: int = 0,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if is_cross:
            self.attnstore(attention_probs, is_cross, self.place_in_unet, region_idx)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CrossAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, attnstore, place_in_unet):
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CrossAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        region_idx: int = 0,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.reshape(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.reshape(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.reshape(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if is_cross:
            query = query.reshape(batch_size * attn.heads, -1, head_dim)
            key = key.reshape(batch_size * attn.heads, -1, head_dim)
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size * attn.heads, -1, attention_mask.shape[-1])
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            self.attnstore(attention_probs, is_cross, self.place_in_unet, region_idx)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def register_attention_control(model, controller, num_regions=1):

    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        if name.endswith("attn2.processor"):  # cross-attention
            cross_att_count += 1
        # # pytorch 1.0
        # attn_procs[name] = CrossAttnProcessor(
        #     attnstore=controller, place_in_unet=place_in_unet
        # )
        # pytorch 2.0
        attn_procs[name] = CrossAttnProcessor2_0(
            attnstore=controller, place_in_unet=place_in_unet
        )

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count * num_regions


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.low_resource else 0

    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str, region_idx: int):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, region_idx: int):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.low_resource:
                attn = self.forward(attn, is_cross, place_in_unet, region_idx)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet, region_idx)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, low_resource=False):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.low_resource = low_resource


class EmptyControl(AttentionControl):

    def forward (self, attn, is_cross: bool, place_in_unet: str, region_idx: int):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str, region_idx: int):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # if attn.shape[1] <= 128 ** 2:  # avoid memory overhead
        self.step_store[region_idx][key].append(attn)
        return attn

    def between_steps(self):
        assert self.cur_step > 0 and self.cur_step <= 51
        if self.cur_step > self.start_step and self.cur_step <= self.end_step:
            if len(self.attention_store[0]) == 0:
                self.attention_store = self.step_store
            else:
                for region_idx in range(self.num_regions):
                    for key in self.attention_store[region_idx]:
                        for i in range(len(self.attention_store[region_idx][key])):
                            self.attention_store[region_idx][key][i] += self.step_store[region_idx][key][i]
        self.step_store = [self.get_empty_store() for _ in range(self.num_regions)]

    def get_average_attention(self):
        average_attention = [{key: [item / (self.end_step - self.start_step) for item in self.attention_store[region_idx][key]] for key in self.attention_store[region_idx]} for region_idx in range(self.num_regions)]
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = [self.get_empty_store() for _ in range(self.num_regions)]
        self.attention_store = [{} for _ in range(self.num_regions)]

    def __init__(self, start_step=0, end_step=51, num_regions=1, low_resource=False):
        super(AttentionStore, self).__init__(low_resource)
        self.step_store = [self.get_empty_store() for _ in range(num_regions)]
        self.attention_store = [{} for _ in range(num_regions)]
        self.start_step = start_step
        self.end_step = end_step
        self.num_regions = num_regions


def save_image(img, save_path):
    pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img.save(save_path)


def aggregate_attention(prompts,
                        attention_store: AttentionStore,
                        res: Tuple[int, int],
                        from_where: List[str],
                        is_cross: bool,
                        select: int,
                        region_idx: int = 0):
    """Aggregates the attention across different layers and heads at the specified resolution."""
    out = []
    attention_maps = attention_store.get_average_attention()[region_idx]
    num_pixels = res[0] * res[1]
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res[0], res[1], item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def get_cross_attention_mask(prompts,
                             attention_store: AttentionStore,
                             height: int,
                             width: int,
                             res: List[Tuple[int, int]],
                             from_where: List[str],
                             threshold_type: str = 'otsu',
                             threshold: float = 0.3,
                             select: int = 0,
                             save_path: str = '',
                             verbose: bool = False,
                             token_ind: int = 1,
                             region_idx: int = 0):
    images = []
    masks = []
    attention_maps = [aggregate_attention(
        prompts, attention_store, r, from_where, True, select, region_idx) for r in res]
    image = [attention_map[:, :, token_ind] for attention_map in attention_maps]
    image = [img.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1) for img in image]  # NxCxHxW
    image = [img / img.max(2, keepdims=True)[0].max(3, keepdims=True)[0] for img in image]  # normalize within 0-1
    image = [img.squeeze(0).permute(1, 2, 0) for img in image]  # HxWxC
    image = [(255 * img).numpy().astype(np.uint8) for img in image]
    image = [np.array(Image.fromarray(img).resize((width, height))) / 255 for img in image]
    image = sum(image) / len(image)
    image = 255 * image / image.max()
    image = image[:, :, 0].astype(np.uint8)  # HxW
    if threshold_type == 'simple':
        mask = np.where((image / 255) > threshold, 1, 0)
    elif threshold_type == 'otsu':
        mask = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        raise NotImplementedError
    if verbose:
        save_image(
            image,
            save_path=os.path.join(save_path, f'region{region_idx}_attn_avg.jpg'))
        save_image(
            255 * mask,
            save_path=os.path.join(save_path, f'region{region_idx}_mask_avg.jpg'))
    images.append(image)
    masks.append(mask)
    return images, masks


def mosaic_coord(center_ratio_range=(0.5, 1.5), img_shape_hw=(512, 512), overlap_hw=(128, 128), num_regions=1, region_option=0):
    # (y1, y2, x1, x2)
    coords = []
    paste_coords = []
    crop_coords = []
    if num_regions == 1:
        # whole image
        coords.append((0, img_shape_hw[0], 0, img_shape_hw[1]))
        paste_coords.append((0, img_shape_hw[0], 0, img_shape_hw[1]))
        crop_coords.append((0, img_shape_hw[0], 0, img_shape_hw[1]))
    elif num_regions == 2:
        if region_option == 0:
            # center x
            center_x = int(
                random.uniform(*center_ratio_range) * img_shape_hw[1]) // 8 * 8
            # left
            coords.append((0, img_shape_hw[0], 0, center_x + overlap_hw[1] // 2))
            paste_coords.append((0, img_shape_hw[0], 0, center_x))
            crop_coords.append((0, img_shape_hw[0], 0, center_x))
            # right
            coords.append((0, img_shape_hw[0], center_x - overlap_hw[1] // 2, img_shape_hw[1] * 2))
            paste_coords.append((0, img_shape_hw[0], center_x, img_shape_hw[1] * 2))
            crop_coords.append((0, img_shape_hw[0], overlap_hw[1] // 2, coords[-1][3] - coords[-1][2]))
        elif region_option == 1:
            # center y
            center_y = int(
                random.uniform(*center_ratio_range) * img_shape_hw[0]) // 8 * 8
            # top
            coords.append((0, center_y + overlap_hw[0] // 2, 0, img_shape_hw[1]))
            paste_coords.append((0, center_y, 0, img_shape_hw[1]))
            crop_coords.append((0, center_y, 0, img_shape_hw[1]))
            # bottom
            coords.append((center_y - overlap_hw[0] // 2, img_shape_hw[0] * 2, 0, img_shape_hw[1]))
            paste_coords.append((center_y, img_shape_hw[0] * 2, 0, img_shape_hw[1]))
            crop_coords.append((overlap_hw[0] // 2, coords[-1][1] - coords[-1][0], 0, img_shape_hw[1]))
        else:
            NotImplementedError
    elif num_regions == 4:
        # mosaic center x, y
        center_x = int(
            random.uniform(*center_ratio_range) * img_shape_hw[1]) // 8 * 8
        center_y = int(
            random.uniform(*center_ratio_range) * img_shape_hw[0]) // 8 * 8
        # top left
        coords.append((0, center_y + overlap_hw[0] // 2, 0, center_x + overlap_hw[1] // 2))
        paste_coords.append((0, center_y, 0, center_x))
        crop_coords.append((0, center_y, 0, center_x))
        # top right
        coords.append((0, center_y + overlap_hw[0] // 2, center_x - overlap_hw[1] // 2, img_shape_hw[1] * 2))
        paste_coords.append((0, center_y, center_x, img_shape_hw[1] * 2))
        crop_coords.append((0, center_y, overlap_hw[1] // 2, coords[-1][3] - coords[-1][2]))
        # bottom left
        coords.append((center_y - overlap_hw[0] // 2, img_shape_hw[0] * 2, 0, center_x + overlap_hw[1] // 2))
        paste_coords.append((center_y, img_shape_hw[0] * 2, 0, center_x))
        crop_coords.append((overlap_hw[0] // 2, coords[-1][1] - coords[-1][0], 0, center_x))
        # bottom right
        coords.append((center_y - overlap_hw[0] // 2, img_shape_hw[0] * 2, center_x - overlap_hw[1] // 2, img_shape_hw[1] * 2))
        paste_coords.append((center_y, img_shape_hw[0] * 2, center_x, img_shape_hw[1] * 2))
        crop_coords.append((overlap_hw[0] // 2, coords[-1][1] - coords[-1][0], overlap_hw[1] // 2, coords[-1][3] - coords[-1][2]))
    else:
        raise NotImplementedError
    return coords, paste_coords, crop_coords
