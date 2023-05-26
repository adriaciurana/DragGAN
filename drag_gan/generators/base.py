from typing import Optional, Union

import torch
from PIL import Image


class BaseGenerator:
    def __init__(
        self,
        features_extractor_size: int,
        default_params: Optional[dict] = None,
    ) -> None:
        self.features_extractor_size = features_extractor_size

        if default_params is None:
            default_params = {}
        self.params = default_params

    @classmethod
    def load_from_pretrained(
        cls,
        name: str,
        device: Union[torch.device, str] = torch.device("cuda:0"),
    ):
        raise NotImplementedError("The load_from_pretrained method has to be defined.")

    def load_from_path(self, network_pkl: str):
        raise NotImplementedError("The load_from_path method has to be defined.")

    def features(self, latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The features method has to be defined.")

    def generate(self, latent: torch.Tensor) -> Image:
        raise NotImplementedError("The generate method has to be defined.")

    def get_latent_from_seed(self, seed: int) -> torch.Tensor:
        raise NotImplementedError("The get_latent_from_seed method has to be defined.")

    def project(self, img: Image, trainable_w_dims: int, *args, **kargs) -> torch.Tensor:
        raise NotImplementedError("The project method has to be defined.")

    def get_gradio_panel(self):
        raise NotImplementedError("This generator is not compatible with gradio.")
