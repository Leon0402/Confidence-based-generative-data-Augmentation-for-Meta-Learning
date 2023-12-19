""" Helper functions to use in the ingestion program. 

AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
"""
import torch
from typing import List


def get_torch_gpu_environment() -> List[str]:
    """ Retrieve all the information regarding the GPU environment.

    Returns:
        List[str]: Information of the GPU environment.
    """
    env_info = list()
    env_info.append(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        env_info.append(f"Cuda version: {torch.version.cuda}")
        env_info.append(f"cuDNN version: {torch.backends.cudnn.version()}")
        env_info.append("Number of available GPUs: " + f"{torch.cuda.device_count()}")
        env_info.append("Current GPU name: " + f"{torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        env_info.append("Number of available GPUs: 0")

    return env_info
