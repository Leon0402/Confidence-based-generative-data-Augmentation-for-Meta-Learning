__all__ = ["GenerativeAugmentation"]

import cv2
import torch
import numpy as np
from PIL import Image
import cdmetadl.dataset
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler, StableDiffusionUpscalePipeline
import diffusers
from .augmentation import Augmentation


class GenerativeAugmentation(Augmentation):

    def __init__(
        self, threshold: float, scale: int, keep_original_data: bool,
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
        controlnet = ControlNetModel.from_pretrained(diffusion_model_id, torch_dtype=torch.float16)
        self.diffusion_model_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            pipeline_model_id, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None,
            requires_safety_checker=False
        ).to(device)

        self.diffusion_model_pipeline.scheduler = DDIMScheduler.from_config(
            self.diffusion_model_pipeline.scheduler.config
        )
        self.diffusion_model_pipeline.enable_model_cpu_offload()
        self.diffusion_model_pipeline.enable_xformers_memory_efficient_attention()
        self.diffusion_model_pipeline.set_progress_bar_config(disable=True)

        diffusers.utils.logging.set_verbosity(40)

    def _init_augmentation(self, support_set: cdmetadl.dataset.SetData,
                           conf_scores: list[float]) -> tuple[cdmetadl.dataset.SetData, None]:
        """
        Abstract method to initialize augmentation-specific parameters.

        :param support_set: The support set.
        :param conf_scores: Confidence scores for each class.
        :return: Specific initialization arguments for augmentation.
        """
        return support_set, None

    def _augment_class(self, cls: int, support_set: cdmetadl.dataset.SetData, number_of_shots: int,
                       init_args: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Augments data for a specific class using the defined image transformations. 
        Used in base class `augment` function.

        :param cls: Class index to augment.
        :param support_set: The support set to augment.
        :param number_of_shots: Number of augmented shots to generate.
        :param init_args: Arguments returned by the `_init_augmentation` function. 
        :return: tuple of the augmented data and labels for the specified class.
        """
        class_name = support_set.class_names[cls]
        random_indices = np.random.randint(0, support_set.number_of_shots, size=number_of_shots)

        diffusion_images = torch.stack([
            self.generate_image(support_set.images_by_class[cls][idx], class_name) for idx in random_indices
        ]).float()
        diffusion_labels = torch.full(size=(number_of_shots, ), fill_value=cls)

        return diffusion_images, diffusion_labels

    def generate_image(self, image, class_name: str):
        image_array = (image * 255).detach().cpu().numpy().astype(np.uint8)
        image_array = np.transpose(image_array, (1, 2, 0))
        image_array = Image.fromarray(image_array)

        upscaled_image = self.upscale_image(image_array)
        edge_image = self.edge_detection(upscaled_image)
        diffusion_image = self.generate_diffusion_image(edge_image, image_class=class_name)
        downscaled_diffusion_image = self.downscale_image(diffusion_image)
        # if True:  #TODO: Delete this if not needed anymore
        #     image_array.save("/home/workstation/Schreibtisch/test_normal.png")
        #     upscaled_image.save("/home/workstation/Schreibtisch/test_upsclaed_normal.png")
        #     edge_image.save("/home/workstation/Schreibtisch/test_edges.png")
        #     diffusion_image.save("/home/workstation/Schreibtisch/test_diffusion.png")
        #     downscaled_diffusion_image.save("/home/workstation/Schreibtisch/test_diffusion_downscaled.png")

        diffusion_array = np.array(downscaled_diffusion_image) / 255
        diffusion_array = torch.tensor(np.transpose(diffusion_array, (2, 0, 1)))

        return diffusion_array

    def upscale_image(self, image_array):
        return image_array.resize((512, 512))

    def downscale_image(self, image):
        return image.resize((128, 128))

    def edge_detection(self, image):
        image = np.array(image)  #512x512x3
        low_threshold = 100
        high_threshold = 200

        canny_image = cv2.Canny(image, low_threshold, high_threshold)  #512x512
        canny_image = canny_image[:, :, None]  #512x512x1
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)  #512x512x3
        canny_image = Image.fromarray(canny_image)

        return canny_image

    def generate_diffusion_image(self, image, image_class: str):
        PROMPT = image_class
        NEGATIVE_PROMPT = "Amputee, Autograph, Bad anatomy, Bad illustration, Bad proportions, Beyond the borders, Blank background, Blurry, Body out of frame, Boring background, Branding, Cropped, Cut off, Deformed, Disfigured, Dismembered, Disproportioned, Distorted, Draft, Duplicate, Duplicated features, Extra arms, Extra fingers, Extra hands, Extra legs, Extra limbs, Fault, Flaw, Fused fingers, Grains, Grainy, Gross proportions, Hazy, Identifying mark, Improper scale, Incorrect physiology, Incorrect ratio, Indistinct, Kitsch, Logo, Long neck, Low quality, Low resolution, Macabre, Malformed, Mark, Misshapen, Missing arms, Missing fingers, Missing hands, Missing legs, Mistake, Morbid, Mutated hands, Mutation, Mutilated, Off-screen, Out of frame, Outside the picture, Pixelated, Poorly drawn face, Poorly drawn feet, Poorly drawn hands, Printed words, Render, Repellent, Replicate, Reproduce, Revolting dimensions, Script, Shortened, Sign, Signature, Split image, Squint, Storyboard, Text, Tiling, Trimmed, Ugly, Unfocused, Unattractive, Unnatural pose, Unreal engine, Unsightly, Watermark, Written language"

        # Run image generation
        return self.diffusion_model_pipeline(
            prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT, image=image, height=512, width=512, num_images_per_prompt=1,
            num_inference_steps=50
        )[0][0]
