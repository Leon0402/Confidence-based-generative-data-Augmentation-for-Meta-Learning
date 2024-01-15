__all__ = ["GenerativeAugmentation"]

import cv2
import torch
import numpy as np
from PIL import Image
import cdmetadl.dataset
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionUpscalePipeline
import diffusers
from .augmentation import Augmentation


class GenerativeAugmentation(Augmentation):

    def __init__(
        self, threshold: float, scale: int, keep_original_data: bool,
        upscaler_id: str = "stabilityai/stable-diffusion-x4-upscaler",
        diffusion_model_id: str = "lllyasviel/sd-controlnet-canny",
        pipeline_model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda"
    ):
        """
        Initializes the StandardAugmentation class with specified threshold, scale, and keep_original_data flags,
        along with a defined set of image transformations.

        Args:
            threshold (float): A threshold value for deciding which classes to augment.
            scale (int): A scale factor for deciding how many samples per classes should be created.
            keep_original_data (bool): A flag to determine whether original data should be included together with the augmented data.
        """
        super().__init__(threshold, scale, keep_original_data)
        print("\n")
        self.upscaler_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            upscaler_id, variant="fp16", torch_dtype=torch.float16
        ).to(device)

        self.upscaler_pipeline.enable_model_cpu_offload()
        self.upscaler_pipeline.enable_xformers_memory_efficient_attention()
        self.upscaler_pipeline.set_progress_bar_config(disable=True)

        controlnet = ControlNetModel.from_pretrained(diffusion_model_id, torch_dtype=torch.float16)
        self.diffusion_model_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            pipeline_model_id, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None,
            requires_safety_checker=False
        ).to(device)

        self.diffusion_model_pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.diffusion_model_pipeline.scheduler.config
        )
        self.diffusion_model_pipeline.enable_model_cpu_offload()
        self.diffusion_model_pipeline.enable_xformers_memory_efficient_attention()
        self.diffusion_model_pipeline.set_progress_bar_config(disable=True)

        diffusers.utils.logging.set_verbosity(40)

    def _init_augmentation(self, support_set: cdmetadl.dataset.SetData, conf_scores: list[float]) -> tuple:
        return None

    def _augment_class(self, cls: int, number_of_shots: int, init_args: list,
                       specific_init_args: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Augments data for a specific class using the defined image transformations. 
        Used in base class `augment` function,

        Args:
            cls (int): The class index for which the data augmentation is to be performed.
            number_of_shots (int): The number of samples to generate.
            init_args (list): General init args of the augmentation like the support_data
            specific_init_args (list): Class specific init args created in the `_init_augmentation` function, 

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the augmented data and corresponding labels for the specified class.
        """
        support_data, _, num_shots_support_set = init_args

        random_indices = np.random.randint(0, num_shots_support_set, size=number_of_shots)

        diffusion_images = torch.stack([self.generateImage(support_data[cls][idx]) for idx in random_indices]).float()
        diffusion_labels = torch.full(size=(number_of_shots, ), fill_value=cls)

        return diffusion_images, diffusion_labels

    def generateImage(self, image):
        image_array = (image * 255).detach().cpu().numpy().astype(np.uint8)
        image_array = np.transpose(image_array, (1, 2, 0))
        image_array = Image.fromarray(image_array)

        upscaled_image = self.upscaleImage(image_array)
        edge_image = self.edgeDetection(upscaled_image)
        diffusion_image = self.generateDiffusionImage(
            edge_image, image_class=""
        )  #TODO: Insert the class name as prompt here
        downscaled_diffusion_image = self.downscaleImage(diffusion_image)
        if False:  #TODO: Delete this if not needed anymore
            image_array.save("/home/workstation/Schreibtisch/test_normal.png")
            edge_image.save("/home/workstation/Schreibtisch/test_edges.png")
            diffusion_image.save("/home/workstation/Schreibtisch/test_diffusion.png")
            downscaled_diffusion_image.save("/home/workstation/Schreibtisch/test_diffusion_downscaled.png")

        diffusion_array = np.array(downscaled_diffusion_image) / 255
        diffusion_array = torch.tensor(np.transpose(diffusion_array, (2, 0, 1)))

        return diffusion_array

    def upscaleImage(self, image_array):
        return self.upscaler_pipeline(prompt="", image=image_array).images[0]

    def downscaleImage(self, image):
        return image.resize((128, 128))

    def edgeDetection(self, image):
        image = np.array(image)
        low_threshold = 100
        high_threshold = 200

        canny_image = cv2.Canny(image, low_threshold, high_threshold)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        canny_image = Image.fromarray(canny_image)

        return canny_image

    def generateDiffusionImage(self, image, image_class: str):
        PROMPT = image_class
        NEGATIVE_PROMPT = "Amputee, Autograph, Bad anatomy, Bad illustration, Bad proportions, Beyond the borders, Blank background, Blurry, Body out of frame, Boring background, Branding, Cropped, Cut off, Deformed, Disfigured, Dismembered, Disproportioned, Distorted, Draft, Duplicate, Duplicated features, Extra arms, Extra fingers, Extra hands, Extra legs, Extra limbs, Fault, Flaw, Fused fingers, Grains, Grainy, Gross proportions, Hazy, Identifying mark, Improper scale, Incorrect physiology, Incorrect ratio, Indistinct, Kitsch, Logo, Long neck, Low quality, Low resolution, Macabre, Malformed, Mark, Misshapen, Missing arms, Missing fingers, Missing hands, Missing legs, Mistake, Morbid, Mutated hands, Mutation, Mutilated, Off-screen, Out of frame, Outside the picture, Pixelated, Poorly drawn face, Poorly drawn feet, Poorly drawn hands, Printed words, Render, Repellent, Replicate, Reproduce, Revolting dimensions, Script, Shortened, Sign, Signature, Split image, Squint, Storyboard, Text, Tiling, Trimmed, Ugly, Unfocused, Unattractive, Unnatural pose, Unreal engine, Unsightly, Watermark, Written language"

        # Run image generation
        return self.diffusion_model_pipeline(
            prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT, image=image, height=512, width=512, num_images_per_prompt=1,
            num_inference_steps=50
        )[0][0]
