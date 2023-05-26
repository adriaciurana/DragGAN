import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch import nn

from drag_gan.utils import TrainableLatent, is_local

from .base import BaseGenerator

CURR_PATH = Path(__file__).parent
sys.path.append(str((CURR_PATH / "../../stylegan2-ada-pytorch").absolute()))  # noqa
import dnnlib  # noqa
import legacy  # noqa
from projector import project  # noqa

feature_extractor_default_callback = lambda res, G: getattr(G.synthesis, f"b{res}")  # noqa


class StyleGANv2Generator(BaseGenerator):
    PRETRAINED_MODELS = {
        "afhqwild": {
            "url": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl",
            "features_extractor_layer": feature_extractor_default_callback,
            "features_extractor_size": 256,
        },
        "afhqcat": {
            "url": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl",
            "features_extractor_layer": feature_extractor_default_callback,
            "features_extractor_size": 256,
        },
        "afhqdog": {
            "url": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl",
            "features_extractor_layer": feature_extractor_default_callback,
            "features_extractor_size": 256,
        },
        "brecahad": {
            "url": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/brecahad.pkl",
            "features_extractor_layer": feature_extractor_default_callback,
            "features_extractor_size": 256,
        },
        "cifar10": {
            "url": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl",
            "features_extractor_layer": feature_extractor_default_callback,
            "features_extractor_size": 16,
        },
        "ffhq": {
            "url": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
            "features_extractor_layer": feature_extractor_default_callback,
            "features_extractor_size": 256,
        },
        "metfaces": {
            "url": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl",
            "features_extractor_layer": feature_extractor_default_callback,
            "features_extractor_size": 256,
        },
    }

    class StyleGANv2TrainableLatent(TrainableLatent):
        pass

    @classmethod
    def load_from_pretrained(
        cls,
        name: str,
        device: Union[torch.device, str] = torch.device("cuda:0"),
    ):
        pretrained_model = cls.PRETRAINED_MODELS[name]
        url_pkl = pretrained_model["url"]
        features_extractor_layer = pretrained_model["features_extractor_layer"]
        features_extractor_size = pretrained_model["features_extractor_size"]

        return cls(
            network_pkl=url_pkl,
            features_extractor_layer=features_extractor_layer,
            features_extractor_size=features_extractor_size,
            device=device,
        )

    def __init__(
        self,
        network_pkl: Optional[str] = None,
        features_extractor_layer: Callable[[int, nn.Module], nn.Module] = feature_extractor_default_callback,
        features_extractor_size: int = 256,
        device: Union[torch.device, str] = torch.device("cuda:0"),
    ):
        super().__init__(
            features_extractor_size=features_extractor_size,
            default_params={"truncation_psi": 0.85, "trainable_w_dims": 6, "trainable_w_dims_axis": 1},
        )
        self._features_extractor_layer = partial(features_extractor_layer, features_extractor_size)
        self.device = device

        if network_pkl is not None:
            self.load_from_path(network_pkl)

    def load_from_path(self, network_pkl: str):
        # Load model ckpt
        if is_local(network_pkl):
            with open(network_pkl, "rb") as f:
                self._G = legacy.load_network_pkl(f)["G_ema"].to(self.device)  # type: ignore

        else:
            with dnnlib.util.open_url(network_pkl) as f:
                self._G = legacy.load_network_pkl(f)["G_ema"].to(self.device)  # type: ignore

        self.input_size = self._G.synthesis.img_resolution

        # If the device is cpu, we need to remap some layers to make it compatible (remove the half precision layers)
        if self.device.type == "cpu":
            self._G = self._G.float()
            for _, module in self._G.named_modules():
                if hasattr(module, "use_fp16"):
                    module.use_fp16 = False

    def features(self, w_latent: torch.Tensor) -> torch.Tensor:
        def forward_layer_hook(F_arr, module, input, output):
            F_arr[0] = output[0]

        F_arr = [None]
        self._features_extractor_layer(self._G).register_forward_hook(partial(forward_layer_hook, F_arr))

        # Features
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            _ = self._G.synthesis(w_latent, noise_mode="const")

        if w_latent.device.type == "cuda":
            torch.cuda.synchronize()

        return F_arr[0]

    def generate(self, w_latent: torch.Tensor) -> Image:
        with torch.no_grad():
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                img = self._G.synthesis(w_latent, noise_mode="const")
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img = Image.fromarray(img[0].cpu().numpy(), "RGB")

        return img

    def get_latent_from_seed(
        self, seed: int, truncation_psi: Optional[int] = None, trainable_w_dims: Optional[int] = None
    ) -> StyleGANv2TrainableLatent:
        if truncation_psi is None:
            truncation_psi = (
                self.params["truncation_psi"] if isinstance(self.params, dict) else self.params.value["truncation_psi"]
            )

        if trainable_w_dims is None:
            trainable_w_dims = (
                self.params["trainable_w_dims"]
                if isinstance(self.params, dict)
                else self.params.value["trainable_w_dims"]
            )

        # Compute first mapping
        with torch.no_grad():
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, self._G.z_dim)).to(self.device)
            c = torch.zeros([1, self._G.c_dim], device=self.device)

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                w_latent = self._G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=None).detach().clone()

        return self.StyleGANv2TrainableLatent(w_latent, trainable_w_dims, dim=1)

    def project(self, img: Image, trainable_w_dims: Optional[int] = None, *args, **kargs) -> StyleGANv2TrainableLatent:
        if trainable_w_dims is None:
            trainable_w_dims = (
                self.params["trainable_w_dims"]
                if isinstance(self.params, dict)
                else self.params.value["trainable_w_dims"]
            )

        # Prepare image for projection
        target_pil = img.convert("RGB")
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((self._G.img_resolution, self._G.img_resolution), Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        target = torch.tensor(target_uint8.transpose([2, 0, 1]), device=self.device)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            projected_w_steps = project(
                G=self._G,
                target=target,
                device=self.device,
                *args,
                **kargs,
            )
        projected_w = projected_w_steps[-1].unsqueeze(0)
        return self.StyleGANv2TrainableLatent(projected_w, trainable_w_dims, dim=1)

    def get_gradio_panel(self, global_state) -> None:
        import gradio as gr

        assert isinstance(global_state, gr.State)
        global_state.value["generator_params"] = self.params

        def on_change_w_latent_dims(trainable_w_dims, global_state):
            global_state["generator_params"]["trainable_w_dims"] = int(trainable_w_dims)
            return global_state

        form_trainable_w_dims_number = gr.Number(
            value=6,
            interactive=True,
            label="Trainable W latent dims",
        ).style(full_width=True)
        form_trainable_w_dims_number.change(
            on_change_w_latent_dims,
            inputs=[form_trainable_w_dims_number, global_state],
            outputs=[global_state],
        )

        def on_change_truncation_psi(truncation_psi, global_state):
            global_state["generator_params"]["truncation_psi"] = truncation_psi
            return global_state

        form_truncation_psi_number = gr.Number(
            value=0.85,
            interactive=True,
            label="Truncation psi",
        ).style(full_width=True)
        form_truncation_psi_number.change(
            on_change_truncation_psi,
            inputs=[form_truncation_psi_number, global_state],
            outputs=[global_state],
        )
