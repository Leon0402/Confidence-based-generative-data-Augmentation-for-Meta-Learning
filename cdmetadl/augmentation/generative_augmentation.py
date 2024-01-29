__all__ = ["GenerativeAugmentation", "Annotator"]

from pathlib import Path
import random

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
import diffusers
from controlnet_aux import ContentShuffleDetector, HEDdetector, MidasDetector, MLSDdetector, CannyDetector, SamDetector

import cdmetadl.dataset

from .augmentation import Augmentation


def set_random_seeds(seed=42, use_cuda=True):
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class Annotator:

    def __init__(
        self, annotator_type: str = "canny", mlsd_value_threshold: float = 0.1, mlsd_distance_threshold: float = 0.1,
        canny_low_threshold: int = 100, canny_high_threshold: int = 200
    ) -> None:
        """
        Defines the annotator and the diffusion model that is used for the edge detection.

        Args: 
            annotator_type (str): String that defines which type of annotator is being used. 
                                Choose one of the following options:
                                - "canny" for Canny edge detector.
                                - "segmentation" for segmentation using UniformerDetector.
                                - "hed" for edge detection using HEDdetector.
                                - "mlsd" for Multi-Scale Line Segment Detector (MLSD).
                                - "midas" for depth estimation using MidasDetector.
            mlsd_value_threshold (float): Threshold value for MLSD (Multi-Scale Line Segment Detector).
            mlsd_distance_threshold (float): Threshold value for MLSD determining distance in the augmentation process.
            canny_low_threshold (int): Lower threshold for Canny edge detector.
            canny_high_threshold (int): Upper threshold for Canny edge detector.
        Return:
            None
        """
        self.annotator_type = annotator_type
        self.model_dict = {
            "canny": ("lllyasviel/control_v11p_sd15_canny", CannyDetector),
            "segmentation": ("lllyasviel/control_v11p_sd15_seg", SamDetector),
            "hed": ("lllyasviel/control_v11p_sd15_softedge", HEDdetector),
            "mlsd": ("lllyasviel/control_v11p_sd15_mlsd", MLSDdetector),
            "midas": ("lllyasviel/control_v11p_sd15_normalbae", MidasDetector),
            "shuffle": ("lllyasviel/control_v11e_sd15_shuffle", ContentShuffleDetector)
        }

        self.mlsd_value_threshold = mlsd_value_threshold
        self.mlsd_distance_threshold = mlsd_distance_threshold
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold

        if not self.annotator_type in self.model_dict.keys():
            raise ValueError(
                f'The annotator "{annotator_type}" is unknown. Please choose one of the following options: {", ".join(self.model_dict.keys())}'
            )

        self.diffusion_model_id, annotator = self.model_dict[annotator_type]
        if annotator_type == "segmentation":
            self.annotator = annotator.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
        elif annotator_type == "canny":
            self.annotator = annotator()
        else:
            self.annotator = annotator.from_pretrained("lllyasviel/Annotators")

    def annotate(self, image: Image.Image) -> Image.Image:
        """
        Annotates an images by using the annotator pipeline that is specified.

        Args:
            image (PIL.Image.Image): Image which edges/features are extrected

        Returns:
            Image.Image: Annotated image
        """
        with torch.no_grad():
            image = np.array(image)
            if self.annotator_type == "mlsd":
                detected_map = self.annotator(image, self.mlsd_value_threshold, self.mlsd_distance_threshold)
            elif self.annotator_type == "canny":
                detected_map = self.annotator(image, self.canny_low_threshold, self.canny_high_threshold)
            else:
                detected_map = self.annotator(image)

            return Image.fromarray(np.array(detected_map))


class GenerativeAugmentation(Augmentation):

    def __init__(
        self, threshold: float, augmentation_size: dict, keep_original_data: bool, annotator_type: str = "canny",
        device: str = "cuda", cache_images: bool = False, mlsd_value_threshold: float = 0.1,
        mlsd_distance_threshold: float = 0.1, canny_low_threshold: int = 100, canny_high_threshold: int = 200
    ) -> None:
        """
        Initializes the StandardAugmentation class with specified threshold, scale, and keep_original_data flags,
        along with a defined set of image transformations.

        Args:
            threshold (float): A threshold value for deciding which classes to augment.
            augmentation_size (dict): Uses for calculation how many shots should be augmented.
            keep_original_data (bool): A flag to determine whether original data should be included together with the augmented data.
            annotator_type (str): String that defines which type of annotator is being used. 
                                Choose one of the following options:
                                - "canny" for Canny edge detector.
                                - "segmentation" for segmentation using UniformerDetector.
                                - "hed" for edge detection using HEDdetector.
                                - "mlsd" for Multi-Scale Line Segment Detector (MLSD).
                                - "midas" for depth estimation using MidasDetector.
            device (str): Defines which device is used for torch operations  
            cache_images (bool): Indicates if the images should be stored in the home-directory for testing purposes
            mlsd_value_threshold (float): Threshold value for MLSD (Multi-Scale Line Segment Detector).
            mlsd_distance_threshold (float): Threshold value for MLSD determining distance in the augmentation process.
            canny_low_threshold (int): Lower threshold for Canny edge detector.
            canny_high_threshold (int): Upper threshold for Canny edge detector.
        Returns:
            None
        """
        generator = torch.Generator(device=device).manual_seed(42)

        super().__init__(threshold, augmentation_size, keep_original_data)
        self.annotator = Annotator(
            annotator_type=annotator_type, mlsd_value_threshold=mlsd_value_threshold,
            mlsd_distance_threshold=mlsd_distance_threshold, canny_low_threshold=canny_low_threshold,
            canny_high_threshold=canny_high_threshold
        )
        controlnet = ControlNetModel.from_pretrained(
            self.annotator.diffusion_model_id, torch_dtype=torch.float16, generator=generator
        )
        self.diffusion_model_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None,
            requires_safety_checker=False
        ).to(device)

        self.diffusion_model_pipeline.scheduler = DDIMScheduler.from_config(
            self.diffusion_model_pipeline.scheduler.config, generator=generator
        )
        self.diffusion_model_pipeline.enable_model_cpu_offload()
        self.diffusion_model_pipeline.enable_xformers_memory_efficient_attention()
        self.diffusion_model_pipeline.set_progress_bar_config(disable=True)
        diffusers.utils.logging.set_verbosity(40)

        # These prompts provide a brief guide using keywords for the desired artistic direction.
        self.style_prompts = [
            "Serenity, Calm, Soft", "Energetic, Dynamic, Movement", "Abstract, Intriguing",
            "Dreamy, Ethereal, Soft Tones", "Bold, Striking, High Contrast", "Harmonious, Balanced",
            "Mystery, Intrigue", "Vibrant, Lively, Colorful", "Minimalist, Clean", "Chaotic, Frenetic",
            "Sophisticated, Elegant", "Nostalgic, Vintage", "Futuristic, Avant-Garde", "Dreamlike, Fantastical",
            "Calming, Peaceful", "Playful, Whimsical", "Timeless, Classic Beauty", "Engaging, Attention-Grabbing",
            "Mysterious, Atmospheric", "Wonder, Curiosity", "Impactful, Memorable", "Surreal Landscape",
            "Nostalgia, Sentimentality", "Freedom, Openness", "Stimulating, Thought-Provoking",
            "Tranquil, Introspective", "Movement, Flow", "Vibrant, Color Focus", "Modern, Contemporary",
            "Unity, Connection"
        ]

        self.cache_images = cache_images
        if self.cache_images:
            self.generated_images = []
        set_random_seeds(seed=42, use_cuda=True)

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
        random_indices = np.random.randint(0, support_set.number_of_shots, size=number_of_shots)

        diffusion_images = torch.stack([
            self.generate_image(support_set.images_by_class[cls][idx])
            for idx in tqdm(random_indices, leave=False, desc=f"Generated Images of class {cls}")
        ]).float()

        diffusion_labels = torch.full(size=(number_of_shots, ), fill_value=cls)

        return diffusion_images, diffusion_labels

    def generate_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Generates a new image using the augmentation pipeline:

        Args:
            image (torch.Tensor): Tensor of the image that will be used as input for the diffusion model
        Returns:
            torch.Tensor: Generated image
        """
        image_array = (image * 255).detach().cpu().numpy().astype(np.uint8)
        image_array = np.transpose(image_array, (1, 2, 0))
        image_array = Image.fromarray(image_array)

        upscaled_image = image_array.resize((512, 512))
        edge_image = self.annotator.annotate(upscaled_image)
        diffusion_image = self.generate_diffusion_image(edge_image)
        downscaled_diffusion_image = diffusion_image.resize((128, 128))
        diffusion_array = np.array(downscaled_diffusion_image) / 255
        diffusion_array = torch.tensor(np.transpose(diffusion_array, (2, 0, 1)))

        if self.cache_images:
            home_dir = Path.home()
            image_array.save(home_dir / "test_normal.png")
            edge_image.save(home_dir / "test_edges.png")
            diffusion_image.save(home_dir / "test_diffusion.png")

            self.generated_images.append({
                "original_image": image_array.resize((512, 512)),
                "edge_map": edge_image.resize((512, 512)),
                "generated_image": diffusion_image.resize((512, 512))
            })

        return diffusion_array

    def generate_diffusion_image(self, image: Image.Image) -> Image.Image:
        """
        Feeds the feature maps/edge map to the diffusion model and generates a new image.

            Args:
                image (PIL.Image.Image): Feature map will be used as input for the diffusion model
        
            Returns:
                Image.Image: Generated image
        """
        POSITIVE_PROMPT = str(np.random.choice(self.style_prompts))
        NEGATIVE_PROMPT = "Amputee, Autograph, Bad anatomy, Bad illustration, Bad proportions, Beyond the borders, Blank background, Blurry, Body out of frame, Boring background, Branding, Cropped, Cut off, Deformed, Disfigured, Dismembered, Disproportioned, Distorted, Draft, Duplicate, Duplicated features, Extra arms, Extra fingers, Extra hands, Extra legs, Extra limbs, Fault, Flaw, Fused fingers, Grains, Grainy, Gross proportions, Hazy, Identifying mark, Improper scale, Incorrect physiology, Incorrect ratio, Indistinct, Kitsch, Logo, Long neck, Low quality, Low resolution, Macabre, Malformed, Mark, Misshapen, Missing arms, Missing fingers, Missing hands, Missing legs, Mistake, Morbid, Mutated hands, Mutation, Mutilated, Off-screen, Out of frame, Outside the picture, Pixelated, Poorly drawn face, Poorly drawn feet, Poorly drawn hands, Printed words, Render, Repellent, Replicate, Reproduce, Revolting dimensions, Script, Shortened, Sign, Signature, Split image, Squint, Storyboard, Text, Tiling, Trimmed, Ugly, Unfocused, Unattractive, Unnatural pose, Unreal engine, Unsightly, Watermark, Written language"

        # Run image generation
        return self.diffusion_model_pipeline(
            prompt=POSITIVE_PROMPT, negative_prompt=NEGATIVE_PROMPT, image=image, height=512, width=512,
            num_images_per_prompt=1, num_inference_steps=50
        )[0][0]
