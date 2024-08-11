import inspect
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from tqdm.auto import tqdm
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from numpy import pi, exp, sqrt
from torchvision.transforms.functional import resize
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers, DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    randn_tensor,
)


class MaskModes(Enum):
    """Modes in which the influence of diffuser is masked"""
    CONSTANT = "constant"
    GAUSSIAN = "gaussian"
    QUARTIC = "quartic"  # see https://en.wikipedia.org/wiki/Kernel_(statistics)


class RerollModes(Enum):
    """Modes in which the reroll regions operate"""
    RESET = "reset"  # completely reset the random noise in the region
    EPSILON = "epsilon"  # alter slightly the latents in the region


@dataclass
class CanvasRegion:
    """Class defining a rectangular region in the canvas"""
    row_init: int  # region starting row in pixel space (included)
    row_end: int  # region end row in pixel space (not included)
    col_init: int  # region starting column in pixel space (included)
    col_end: int  # region end column in pixel space (not included)
    region_seed: int = None  # seed for random operations in this region
    noise_eps: float = 0.0  # deviation of a zero-mean gaussian noise to be applied over the latents in this region, useful for slightly "rerolling" latents

    def __post_init__(self):
        # initialize arguments if not specified
        if self.region_seed is None:
            self.region_seed = np.random.randint(9999999999)
        # check coordinates are non-negative
        for coord in [self.row_init, self.row_end, self.col_init, self.col_end]:
            if coord < 0:
                raise ValueError(f"A CanvasRegion must be defined with non-negative indices, found ({self.row_init}, {self.row_end}, {self.col_init}, {self.col_end})")
        # check coordinates are divisible by 8, else we end up with nasty rounding error when mapping to latent space
        for coord in [self.row_init, self.row_end, self.col_init, self.col_end]:
            if coord // 8 != coord / 8:
                raise ValueError(f"A CanvasRegion must be defined with locations divisible by 8, found ({self.row_init}-{self.row_end}, {self.col_init}-{self.col_end})")
        # check noise eps is non-negative
        if self.noise_eps < 0:
            raise ValueError(f"A CanvasRegion must be defined noises eps non-negative, found {self.noise_eps}")
        # compute coordinates for this region in latent space
        self.latent_row_init = self.row_init // 8
        self.latent_row_end = self.row_end // 8
        self.latent_col_init = self.col_init // 8
        self.latent_col_end = self.col_end // 8

    @property
    def width(self):
        return self.col_end - self.col_init

    @property
    def height(self):
        return self.row_end - self.row_init

    def get_region_generator(self, device="cpu"):
        """Creates a torch.Generator based on the random seed of this region"""
        # initialize region generator
        return torch.Generator(device).manual_seed(self.region_seed)

    @property
    def __dict__(self):
        return asdict(self)


@dataclass
class DiffusionRegion(CanvasRegion):
    """Abstract class defining a region where some class of diffusion process is acting"""
    pass


@dataclass
class RerollRegion(CanvasRegion):
    """Class defining a rectangular canvas region in which initial latent noise will be rerolled"""
    reroll_mode: RerollModes = RerollModes.RESET.value


@dataclass
class Text2ImageRegion(DiffusionRegion):
    """Class defining a region where a text guided diffusion process is acting"""
    prompt: str = ""  # text prompt guiding the diffuser in this region
    guidance_scale: float = 7.5  # guidance scale of the diffuser in this region, randomize if given as None
    mask_type: MaskModes = MaskModes.GAUSSIAN.value  # kind of weight mask applied to this region
    mask_weight: float = 1.0  # global weights multiplier of the mask
    tokenized_prompt = None  # tokenized prompt
    encoded_prompt = None  # encoded prompt

    def __post_init__(self):
        super().__post_init__()
        # mask weight cannot be negative
        if self.mask_weight < 0:
            raise ValueError(f"A Text2ImageRegion must be defined with non-negative mask weight, found {self.mask_weight}")
        # mask type must be an actual known mask
        if self.mask_type not in [e.value for e in MaskModes]:
            raise ValueError(f"A Text2ImageRegion was defined with mask {self.mask_type}, which is not an accepted mask ({[e.value for e in MaskModes]})")
        # randomize arguments if given as None
        if self.guidance_scale is None:
            self.guidance_scale = np.random.randint(5, 30)
        # clean prompt
        self.prompt = re.sub(' +', ' ', self.prompt).replace("\n", " ")

    def tokenize_prompt(self, tokenizer):
        """Tokenizes the prompt for this diffusion region using a given tokenizer"""
        self.tokenized_prompt = tokenizer(self.prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    def encode_prompt(self, text_encoder, device):
        """Encodes the previously tokenized prompt for this diffusion region using a given encoder"""
        assert self.tokenized_prompt is not None, ValueError("Prompt in diffusion region must be tokenized before encoding")
        self.encoded_prompt = text_encoder(self.tokenized_prompt.input_ids.to(device))[0]


@dataclass
class Text2ImageRegionXL(DiffusionRegion):
    """Class defining a region where a text guided diffusion process is acting"""
    prompt: str = ""  # text prompt guiding the diffuser in this region
    prompt_2: Optional[str] = None
    negative_prompt: Optional[str] = None
    negative_prompt_2: Optional[str] = None
    guidance_scale: float = 7.5  # guidance scale of the diffuser in this region, randomize if given as None
    mask_type: MaskModes = MaskModes.GAUSSIAN.value  # kind of weight mask applied to this region
    mask_weight: float = 1.0  # global weights multiplier of the mask
    prompt_embeds: Optional[torch.FloatTensor] = None
    negative_prompt_embeds: Optional[torch.FloatTensor] = None
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None

    def __post_init__(self):
        super().__post_init__()
        # mask weight cannot be negative
        if self.mask_weight < 0:
            raise ValueError(f"A Text2ImageRegionXL must be defined with non-negative mask weight, found {self.mask_weight}")
        # mask type must be an actual known mask
        if self.mask_type not in [e.value for e in MaskModes]:
            raise ValueError(f"A Text2ImageRegionXL was defined with mask {self.mask_type}, which is not an accepted mask ({[e.value for e in MaskModes]})")
        # randomize arguments if given as None
        if self.guidance_scale is None:
            self.guidance_scale = np.random.randint(5, 30)
        # clean prompt
        self.prompt = re.sub(' +', ' ', self.prompt).replace("\n", " ")
        if self.prompt_2 is not None:
            self.prompt_2 = re.sub(' +', ' ', self.prompt_2).replace("\n", " ")
        if self.negative_prompt is not None:
            self.negative_prompt = re.sub(' +', ' ', self.negative_prompt).replace("\n", " ")
        if self.negative_prompt_2 is not None:
            self.negative_prompt_2 = re.sub(' +', ' ', self.negative_prompt_2).replace("\n", " ")
        
    def encode_prompt(self, tokenizers, text_encoders, device, config):
        """Tokenize and encode the prompt for this diffusion region using a given tokenizer and encoder"""
        self.prompt_2 = self.prompt_2 or self.prompt
        prompt_embeds_list = []
        prompts = [self.prompt, self.prompt_2]
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
            # we are only ALWAYS interested in the pooled output of the final text encoder
            self.pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)
        self.prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        zero_out_negative_prompt = config.force_zeros_for_empty_prompt
        if zero_out_negative_prompt:
            self.negative_prompt_embeds = torch.zeros_like(self.prompt_embeds)
            self.negative_pooled_prompt_embeds = torch.zeros_like(self.pooled_prompt_embeds)
        else:
            self.negative_prompt = self.negative_prompt or ""
            self.negative_prompt_2 = self.negative_prompt_2 or self.negative_prompt
            uncond_tokens = [self.negative_prompt, self.negative_prompt_2]
            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                max_length = self.prompt_embeds.shape[1]
                uncond_input = tokenizer(negative_prompt, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
                negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(device), output_hidden_states=True)
                # we are only ALWAYS interested in the pooled output of the final text encoder
                self.negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
                negative_prompt_embeds_list.append(negative_prompt_embeds)
            self.negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)
        self.prompt_embeds = self.prompt_embeds.to(dtype=text_encoders[-1].dtype, device=device)
        self.negative_prompt_embeds = self.negative_prompt_embeds.to(dtype=text_encoders[-1].dtype, device=device)


@dataclass
class Image2ImageRegion(DiffusionRegion):
    """Class defining a region where an image guided diffusion process is acting"""
    reference_image: torch.FloatTensor = None
    strength: float = 0.8  # strength of the image

    def __post_init__(self):
        super().__post_init__()
        if self.reference_image is None:
            raise ValueError("Must provide a reference image when creating an Image2ImageRegion")
        if self.strength < 0 or self.strength > 1:
          raise ValueError(f'The value of strength should in [0.0, 1.0] but is {self.strength}')
        # rescale image to region shape
        self.reference_image = resize(self.reference_image, size=[self.height, self.width])

    def encode_reference_image(self, encoder, device, generator, cpu_vae=False):
        """Encodes the reference image for this Image2Image region into the latent space"""
        # place encoder in CPU or not following the parameter cpu_vae
        if cpu_vae:
            # note here we use mean instead of sample, to avoid moving also generator to CPU, which is troublesome
            self.reference_latents = encoder.cpu().encode(self.reference_image).latent_dist.mean.to(device)
        else:
            self.reference_latents = encoder.encode(self.reference_image.to(device)).latent_dist.sample(generator=generator)
        self.reference_latents = 0.18215 * self.reference_latents

    @property
    def __dict__(self):
        # this class requires special casting to dict because of the reference_image tensor, otherwise it cannot be casted to JSON

        # get all basic fields from parent class
        super_fields = {key: getattr(self, key) for key in DiffusionRegion.__dataclass_fields__.keys()}
        # pack other fields
        return {
            **super_fields,
            "reference_image": self.reference_image.cpu().tolist(),
            "strength": self.strength
        }


@dataclass
class MaskWeightsBuilder:
    """Auxiliary class to compute a tensor of weights for a given diffusion region"""
    latent_space_dim: int  # size of the U-net latent space
    nbatch: int = 1  # batch size in the U-net

    def compute_mask_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Computes a tensor of weights for a given diffusion region"""
        MASK_BUILDERS = {
            MaskModes.CONSTANT.value: self._constant_weights,
            MaskModes.GAUSSIAN.value: self._gaussian_weights,
            MaskModes.QUARTIC.value: self._quartic_weights,
        }
        return MASK_BUILDERS[region.mask_type](region)

    def _constant_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Computes a tensor of constant for a given diffusion region"""
        latent_width = region.latent_col_end - region.latent_col_init
        latent_height = region.latent_row_end - region.latent_row_init
        return torch.ones(self.nbatch, self.latent_space_dim, latent_height, latent_width) * region.mask_weight

    def _gaussian_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Generates a gaussian mask of weights for tile contributions"""
        latent_width = region.latent_col_end - region.latent_col_init
        latent_height = region.latent_row_end - region.latent_row_init

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = (latent_height -1) / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]
        
        weights = np.outer(y_probs, x_probs) * region.mask_weight
        return torch.tile(torch.tensor(weights), (self.nbatch, self.latent_space_dim, 1, 1))

    def _quartic_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Generates a quartic mask of weights for tile contributions
        
        The quartic kernel has bounded support over the diffusion region, and a smooth decay to the region limits.
        """
        quartic_constant = 15. / 16.        

        support = (np.array(range(region.latent_col_init, region.latent_col_end)) - region.latent_col_init) / (region.latent_col_end - region.latent_col_init - 1) * 1.99 - (1.99 / 2.)
        x_probs = quartic_constant * np.square(1 - np.square(support))
        support = (np.array(range(region.latent_row_init, region.latent_row_end)) - region.latent_row_init) / (region.latent_row_end - region.latent_row_init - 1) * 1.99 - (1.99 / 2.)
        y_probs = quartic_constant * np.square(1 - np.square(support))

        weights = np.outer(y_probs, x_probs) * region.mask_weight
        return torch.tile(torch.tensor(weights), (self.nbatch, self.latent_space_dim, 1, 1))
        

class MosaicFusionPipeline(DiffusionPipeline):
    """Pipeline for MosaicFusion using Stable Diffusion"""
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler],
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )

    def decode_latents(self, latents, cpu_vae=False):
        """Decodes a given array of latents into pixel space"""
        # scale and decode the image latents with vae
        if cpu_vae:
            lat = deepcopy(latents).cpu()
            vae = deepcopy(self.vae).cpu()
        else:
            lat = latents
            vae = self.vae

        lat = 1 / 0.18215 * lat
        image = vae.decode(lat).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        
        image = (image * 255).astype(np.uint8)

        return image

    def get_latest_timestep_img2img(self, num_inference_steps, strength):
        """Finds the latest timesteps where an img2img strength does not impose latents anymore"""
        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * (1 - strength)) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        t_start = min(max(num_inference_steps - init_timestep + offset, 0), num_inference_steps-1)
        latest_timestep = self.scheduler.timesteps[t_start]

        return latest_timestep

    @torch.no_grad()
    def __call__(
        self,
        controller,
        canvas_height: int,
        canvas_width: int,
        regions: List[DiffusionRegion],
        num_inference_steps: Optional[int] = 50,
        seed: Optional[int] = 12345,
        low_resource: Optional[bool] = False,
        reroll_regions: Optional[List[RerollRegion]] = None,
        cpu_vae: Optional[bool] = False,
        decode_steps: Optional[bool] = False
    ):
        if reroll_regions is None:
            reroll_regions = []
        batch_size = 1

        if decode_steps:
            steps_images = []

        # prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # split diffusion regions by their kind
        text2image_regions = [region for region in regions if isinstance(region, Text2ImageRegion)]
        image2image_regions = [region for region in regions if isinstance(region, Image2ImageRegion)]

        # prepare text embeddings
        for region in text2image_regions:
            region.tokenize_prompt(self.tokenizer)
            region.encode_prompt(self.text_encoder, self.device)

        # create original noisy latents using the timesteps
        latents_shape = (batch_size, self.unet.config.in_channels, canvas_height // 8, canvas_width // 8)
        generator = torch.Generator(self.device).manual_seed(seed)
        init_noise = torch.randn(latents_shape, generator=generator, device=self.device, dtype=self.text_encoder.dtype)

        # reset latents in seed reroll regions, if requested
        for region in reroll_regions:
            if region.reroll_mode == RerollModes.RESET.value:
                region_shape = (latents_shape[0], latents_shape[1], region.latent_row_end - region.latent_row_init, region.latent_col_end - region.latent_col_init)
                init_noise[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end] = torch.randn(region_shape, generator=region.get_region_generator(self.device), device=self.device, dtype=self.text_encoder.dtype)

        # apply epsilon noise to regions: first diffusion regions, then reroll regions
        all_eps_rerolls = regions + [r for r in reroll_regions if r.reroll_mode == RerollModes.EPSILON.value]
        for region in all_eps_rerolls:
            if region.noise_eps > 0:
                region_noise = init_noise[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end]
                eps_noise = torch.randn(region_noise.shape, generator=region.get_region_generator(self.device), device=self.device, dtype=self.text_encoder.dtype) * region.noise_eps
                init_noise[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end] += eps_noise

        # scale the initial noise by the standard deviation required by the scheduler
        latents = init_noise * self.scheduler.init_noise_sigma

        # get unconditional embeddings for classifier free guidance in text2image regions
        for region in text2image_regions:
            max_length = region.tokenized_prompt.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes.
            if low_resource:
                region.encoded_prompt = [uncond_embeddings, region.encoded_prompt]
            else:
                region.encoded_prompt = torch.cat([uncond_embeddings, region.encoded_prompt])

        # prepare image latents
        for region in image2image_regions:
            region.encode_reference_image(self.vae, device=self.device, generator=generator)

        # prepare mask of weights for each region
        mask_builder = MaskWeightsBuilder(latent_space_dim=self.unet.config.in_channels, nbatch=batch_size)
        mask_weights = [mask_builder.compute_mask_weights(region).to(device=self.device, dtype=self.text_encoder.dtype) for region in text2image_regions]

        # diffusion timesteps
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # diffuse each region            
            noise_preds_regions = []

            # text2image regions
            for region_idx, region in enumerate(text2image_regions):
                region_latents = latents[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end]
                # expand the latents if we are doing classifier free guidance
                latent_model_input = region_latents if low_resource else torch.cat([region_latents] * 2)
                # scale model input following scheduler rules
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # predict the noise residual
                cross_attention_kwargs = {'region_idx': region_idx}
                if low_resource:
                    noise_pred_uncond = self.unet(latent_model_input, t, encoder_hidden_states=region.encoded_prompt[0], cross_attention_kwargs=cross_attention_kwargs)["sample"]
                    noise_pred_text = self.unet(latent_model_input, t, encoder_hidden_states=region.encoded_prompt[1], cross_attention_kwargs=cross_attention_kwargs)["sample"]
                else:
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=region.encoded_prompt, cross_attention_kwargs=cross_attention_kwargs)["sample"]
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # perform guidance
                noise_pred_region = noise_pred_uncond + region.guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_preds_regions.append(noise_pred_region)
                
            # merge noise predictions for all tiles
            noise_pred = torch.zeros(latents.shape, device=self.device, dtype=self.text_encoder.dtype)
            contributors = torch.zeros(latents.shape, device=self.device, dtype=self.text_encoder.dtype)
            # add each tile contribution to overall latents
            for region, noise_pred_region, mask_weights_region in zip(text2image_regions, noise_preds_regions, mask_weights):
                noise_pred[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end] += noise_pred_region * mask_weights_region
                contributors[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end] += mask_weights_region
            # average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            noise_pred = torch.nan_to_num(noise_pred)  # replace NaNs by zeros: NaN can appear if a position is not covered by any DiffusionRegion

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Image2Image regions: override latents generated by the scheduler
            for region in image2image_regions:
                influence_step = self.get_latest_timestep_img2img(num_inference_steps, region.strength)
                # only override in the timesteps before the last influence step of the image (given by its strength)
                if t > influence_step:
                    timestep = t.repeat(batch_size)
                    region_init_noise = init_noise[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end]
                    region_latents = self.scheduler.add_noise(region.reference_latents, region_init_noise, timestep)
                    latents[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end] = region_latents

            if decode_steps:
                steps_images.append(self.decode_latents(latents, cpu_vae))

        # scale and decode the image latents with vae
        image = self.decode_latents(latents, cpu_vae)

        output = {"sample": image}
        if decode_steps:
            output = {**output, "steps_images": steps_images}
        return output


class MosaicFusionXLPipeline(DiffusionPipeline):
    """Pipeline for MosaicFusion using Stable Diffusion XL"""
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = self.unet.config.sample_size

    # copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        model_sequence = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )
        model_sequence.extend([self.unet, self.vae])

        hook = None
        for cpu_offloaded_model in model_sequence:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # we'll offload the last model manually
        self.final_offload_hook = hook

    def decode_latents(self, latents, cpu_vae=False):
        """Decodes a given array of latents into pixel space"""
        # scale and decode the image latents with vae
        if cpu_vae:
            lat = deepcopy(latents).cpu()
            vae = deepcopy(self.vae).cpu()
        else:
            lat = latents
            vae = self.vae

        lat = 1 / 0.18215 * lat
        image = vae.decode(lat).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        
        image = (image * 255).astype(np.uint8)

        return image

    def get_latest_timestep_img2img(self, num_inference_steps, strength):
        """Finds the latest timesteps where an img2img strength does not impose latents anymore"""
        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * (1 - strength)) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        t_start = min(max(num_inference_steps - init_timestep + offset, 0), num_inference_steps-1)
        latest_timestep = self.scheduler.timesteps[t_start]

        return latest_timestep

    # copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @torch.no_grad()
    def __call__(
        self,
        controller,
        canvas_height: int,
        canvas_width: int,
        regions: List[DiffusionRegion],
        num_inference_steps: Optional[int] = 50,
        seed: Optional[int] = 12345,
        low_resource: Optional[bool] = False,
        reroll_regions: Optional[List[RerollRegion]] = None,
        cpu_vae: Optional[bool] = False,
        decode_steps: Optional[bool] = False,
        eta: float = 0.0
    ):
        # define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )
        
        if reroll_regions is None:
            reroll_regions = []
        batch_size = 1

        if decode_steps:
            steps_images = []

        # prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # split diffusion regions by their kind
        text2image_regions = [region for region in regions if isinstance(region, Text2ImageRegionXL)]
        image2image_regions = [region for region in regions if isinstance(region, Image2ImageRegion)]

        # prepare text embeddings
        for region in text2image_regions:
            region.encode_prompt(tokenizers, text_encoders, self.device, self.config)

        # create original noisy latents using the timesteps
        latents_shape = (batch_size, self.unet.config.in_channels, canvas_height // self.vae_scale_factor, canvas_width // self.vae_scale_factor)
        generator = torch.Generator(self.device).manual_seed(seed)
        init_noise = torch.randn(latents_shape, generator=generator, device=self.device, dtype=self.text_encoder_2.dtype)

        # reset latents in seed reroll regions, if requested
        for region in reroll_regions:
            if region.reroll_mode == RerollModes.RESET.value:
                region_shape = (latents_shape[0], latents_shape[1], region.latent_row_end - region.latent_row_init, region.latent_col_end - region.latent_col_init)
                init_noise[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end] = torch.randn(region_shape, generator=region.get_region_generator(self.device), device=self.device, dtype=self.text_encoder_2.dtype)

        # apply epsilon noise to regions: first diffusion regions, then reroll regions
        all_eps_rerolls = regions + [r for r in reroll_regions if r.reroll_mode == RerollModes.EPSILON.value]
        for region in all_eps_rerolls:
            if region.noise_eps > 0:
                region_noise = init_noise[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end]
                eps_noise = torch.randn(region_noise.shape, generator=region.get_region_generator(self.device), device=self.device, dtype=self.text_encoder_2.dtype) * region.noise_eps
                init_noise[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end] += eps_noise

        # scale the initial noise by the standard deviation required by the scheduler
        latents = init_noise * self.scheduler.init_noise_sigma

        # prepare image latents
        for region in image2image_regions:
            region.encode_reference_image(self.vae, device=self.device, generator=generator)

        # prepare mask of weights for each region
        mask_builder = MaskWeightsBuilder(latent_space_dim=self.unet.config.in_channels, nbatch=batch_size)
        mask_weights = [mask_builder.compute_mask_weights(region).to(device=self.device, dtype=self.text_encoder_2.dtype) for region in text2image_regions]

        # prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # diffusion timesteps
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # diffuse each region            
            noise_preds_regions = []

            # text2image regions
            for region_idx, region in enumerate(text2image_regions):
                region_latents = latents[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end]
                # expand the latents if we are doing classifier free guidance
                latent_model_input = region_latents if low_resource else torch.cat([region_latents] * 2)
                # scale model input following scheduler rules
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # prepare added time ids & embeddings
                add_text_embeds = region.pooled_prompt_embeds
                add_time_ids = self._get_add_time_ids(
                    (region.height, region.width), (0, 0), (region.height, region.width), dtype=region.prompt_embeds.dtype
                )
                # predict the noise residual
                cross_attention_kwargs = {'region_idx': region_idx}
                if low_resource:
                    prompt_embeds = [region.negative_prompt_embeds, region.prompt_embeds]
                    add_text_embeds = [region.negative_pooled_prompt_embeds, add_text_embeds]
                    add_time_ids = [add_time_ids, add_time_ids]
                    added_cond_kwargs = {"text_embeds": add_text_embeds[0], "time_ids": add_time_ids[0]}
                    noise_pred_uncond = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds[0], cross_attention_kwargs=cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
                    added_cond_kwargs = {"text_embeds": add_text_embeds[1], "time_ids": add_time_ids[1]}
                    noise_pred_text = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds[1], cross_attention_kwargs=cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
                else:
                    prompt_embeds = torch.cat([region.negative_prompt_embeds, region.prompt_embeds], dim=0).to(self.device)
                    add_text_embeds = torch.cat([region.negative_pooled_prompt_embeds, add_text_embeds], dim=0).to(self.device)
                    add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(self.device)
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs, return_dict=False)[0]
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # perform guidance
                noise_pred_region = noise_pred_uncond + region.guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_preds_regions.append(noise_pred_region)
                
            # merge noise predictions for all tiles
            noise_pred = torch.zeros(latents.shape, device=self.device, dtype=self.text_encoder_2.dtype)
            contributors = torch.zeros(latents.shape, device=self.device, dtype=self.text_encoder_2.dtype)
            # add each tile contribution to overall latents
            for region, noise_pred_region, mask_weights_region in zip(text2image_regions, noise_preds_regions, mask_weights):
                noise_pred[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end] += noise_pred_region * mask_weights_region
                contributors[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end] += mask_weights_region
            # average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            noise_pred = torch.nan_to_num(noise_pred)  # replace NaNs by zeros: NaN can appear if a position is not covered by any DiffusionRegion

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # Image2Image regions: override latents generated by the scheduler
            for region in image2image_regions:
                influence_step = self.get_latest_timestep_img2img(num_inference_steps, region.strength)
                # only override in the timesteps before the last influence step of the image (given by its strength)
                if t > influence_step:
                    timestep = t.repeat(batch_size)
                    region_init_noise = init_noise[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end]
                    region_latents = self.scheduler.add_noise(region.reference_latents, region_init_noise, timestep)
                    latents[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end] = region_latents

            if decode_steps:
                steps_images.append(self.decode_latents(latents, cpu_vae))

        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        
        # scale and decode the image latents with vae
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="np")
        image = (image * 255).round().astype(np.uint8)

        # offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        output = {"sample": image}
        if decode_steps:
            output = {**output, "steps_images": steps_images}
        return output
