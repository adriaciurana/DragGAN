import sys
import warnings
from functools import partial
from os.path import exists
from pathlib import Path
from typing import Callable, Optional, Union
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from PIL import Image
from qqdm import qqdm
from torch import nn, optim
from torch.nn import functional as Functional

CURR_PATH = Path(__file__).parent
sys.path.append(str((CURR_PATH / "stylegan2-ada-pytorch").absolute()))  # noqa
import dnnlib  # noqa
import legacy  # noqa
from projector import project  # noqa


def is_local(url):
    url_parsed = urlparse(url)
    if url_parsed.scheme in ("file", ""):  # Possibly a local file
        return exists(url_parsed.path)
    return False


def fix_point(p: torch.Tensor, input_size: int) -> torch.Tensor:
    scale_factor = 2.0 / input_size
    p_fix = p * scale_factor - 1
    return p_fix


def fix_radius(r: torch.Tensor, input_size: int) -> torch.Tensor:
    scale_factor = 2.0 / input_size
    r_fix = r * scale_factor
    return r_fix


def cdist_p_norm(x: torch.Tensor, y: torch.Tensor, p: int = 1) -> torch.Tensor:
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    y = y.reshape(y.shape[0], y.shape[1], -1).permute(0, 2, 1)

    x = x[:, :, None].expand(-1, -1, y.shape[1], -1)
    y = y[:, None].expand(-1, x.shape[1], -1, -1)
    return torch.norm(x - y, p=p, dim=-1)


def generate_motion_samples(
    p: torch.Tensor,
    r: torch.Tensor,
    samples: int = 5,
) -> torch.Tensor:
    p_exp = p.reshape(p.shape[0], 1, 1, 2)
    q_samples = p_exp.expand(-1, samples, samples, -1)  # N x step x step x 2

    radius_meshgrid = torch.stack(
        [
            torch.stack(
                torch.meshgrid(2 * [torch.linspace(-r_i, r_i, samples)], indexing="ij"),
                dim=-1,
            )
            for r_i in r
        ],
        dim=0,
    ).to(
        q_samples.device
    )  # N x step x step x 2

    q_samples = torch.clip(q_samples + radius_meshgrid, -1.0, 1.0)

    return q_samples


def generate_motion_direction(p: torch.Tensor, t: torch.Tensor, magnitude_direction: float = 0.1) -> torch.Tensor:
    dir_pt = magnitude_direction * Functional.normalize(t - p, dim=-1)
    return dir_pt[:, None, None, :]


def generate_motion_masks(mask_in_pixels, output_size=256, dims=128) -> torch.Tensor:
    return (
        torch.nn.functional.interpolate(
            mask_in_pixels.reshape(1, 1, mask_in_pixels.shape[0], mask_in_pixels.shape[1]).float(),
            size=[output_size, output_size],
            mode="nearest",
        )
        .reshape(1, 1, output_size, output_size)
        .expand(-1, dims, -1, -1)
    )


def draw_p_image(img_pil: Image, p: torch.Tensor, t: torch.Tensor, input_size: int):
    img_cv = np.array(img_pil)
    for p_i, t_i in zip(p, t):
        p_i_pixels = (p_i + 1) / 2.0 * input_size
        t_i_pixels = (t_i + 1) / 2.0 * input_size

        rad_draw = int(input_size * 0.02)
        cv2.circle(img_cv, (int(p_i_pixels[0]), int(p_i_pixels[1])), rad_draw, (255, 0, 0), -1)
        cv2.circle(img_cv, (int(t_i_pixels[0]), int(t_i_pixels[1])), rad_draw, (0, 0, 255), -1)
    return Image.fromarray(img_cv)


class DragGAN:
    def __init__(
        self,
        network_pkl: str,
        features_extractor_layer: Callable[[nn.Module], nn.Module] = lambda G: G.synthesis.b256,
        features_extractor_size: int = 256,
        features_extractor_dims: int = 128,
        device: Union[torch.device, str] = torch.device("cuda:0"),
    ):
        # Load model ckpt
        if is_local(network_pkl):
            with open(network_pkl, "rb") as f:
                self._G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

        else:
            with dnnlib.util.open_url(network_pkl) as f:
                self._G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

        self._input_size = self._G.synthesis.img_resolution
        self._features_extractor_layer = features_extractor_layer
        self._features_extractor_size = features_extractor_size
        self._features_extractor_dims = features_extractor_dims
        self._device = device

    def pixel2norm(self, p: torch.Tensor):
        return fix_point(p, self._input_size)

    def norm2pixel(self, p: torch.Tensor):
        scale_factor = self._input_size / 2.0
        p_fix = (p + 1) * scale_factor
        return p_fix

    def radius2norm(self, r: torch.Tensor):
        return fix_radius(r, self._input_size)

    def _get_F(self, w_latent_learn: torch.Tensor, w_latent_fix: torch.Tensor) -> torch.Tensor:
        def forward_layer_hook(F_arr, module, input, output):
            F_arr[0] = output[0]

        F_arr = [None]
        self._features_extractor_layer(self._G).register_forward_hook(partial(forward_layer_hook, F_arr))

        # Features
        w_latent = torch.cat((w_latent_learn, w_latent_fix), dim=1)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            _ = self._G.synthesis(w_latent, noise_mode="const")

        if w_latent.device.type == "cuda":
            torch.cuda.synchronize()

        return F_arr[0]

    def get_w_latent_from_seed(self, seed: int, truncation_psi: int = 0.85) -> torch.Tensor:
        # Compute first mapping
        with torch.no_grad():
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, self._G.z_dim)).to(self._device)
            c = torch.zeros([1, self._G.c_dim], device=self._device)

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                w_latent_orig = (
                    self._G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=None).detach().clone()
                )

        return w_latent_orig

    def _compute_motion_step(
        self,
        F0: torch.Tensor,
        p: torch.Tensor,
        t: torch.Tensor,
        r1: torch.Tensor,
        M: torch.Tensor,
        w_latent_learn: torch.Tensor,
        w_latent_fix: torch.Tensor,
        optimizer: optim,
        steps: int = 300,
        lambda_value: float = 12.0,
        magnitude_direction: int = 0.1,
        r1_interpolation_samples: int = 3,
        pbar: Optional[qqdm] = None,
    ):
        # Generate masks
        qr1_samples = generate_motion_samples(p, r1, samples=r1_interpolation_samples)
        qr1d_samples = qr1_samples + generate_motion_direction(p, t, magnitude_direction=magnitude_direction)

        # Start training
        for step_idx in range(steps):
            optimizer.zero_grad()

            # Extract the features from the current w_latent
            F = self._get_F(w_latent_learn, w_latent_fix)
            F_exp = F.expand(qr1_samples.shape[0], -1, -1, -1)

            # F(q)
            # Use grid_sample to extract the features for q, where q € neighbours(p, r1)
            Fq = torch.nn.functional.grid_sample(
                F_exp.float(), qr1_samples.float(), mode="bilinear", align_corners=False
            ).detach()

            # F(q + d)
            # Use grid_sample to extract the features for q + d, where q € neighbours(p, r1)
            Fqd = torch.nn.functional.grid_sample(
                F_exp.float(),
                qr1d_samples.float(),
                mode="bilinear",
                align_corners=False,
            )

            loss = Functional.l1_loss(Fq, Fqd)
            if M is not None:
                loss += lambda_value * Functional.l1_loss(F * (1 - M), F0 * (1 - M))

            if pbar is not None:
                info_dict = pbar.info_dict.copy()
                info_dict["loss"] = float(loss)
                info_dict["motion_step"] = f"{step_idx}/{steps}"
                pbar.set_infos(info_dict)
                pbar.update(n=0)

            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def _compute_point_tracking(
        self,
        F0: torch.Tensor,
        w_latent_learn: torch.Tensor,
        w_latent_fix: torch.Tensor,
        p: torch.Tensor,
        p_init: torch.Tensor,
        r2: torch.Tensor,
        t: torch.Tensor,
        r2_interpolation_samples: int = 12,
        distance_l_type: int = 1,
        pbar: Optional[qqdm] = None,
    ):
        # Generate p an q € neighbours(p, r2)
        p_init_samples = p_init.reshape(p_init.shape[0], 1, 1, 2)
        qr2_samples = generate_motion_samples(p, r=r2, samples=r2_interpolation_samples)

        # Extract the features from the current w_latent
        F0_exp = F0.expand(qr2_samples.shape[0], -1, -1, -1)
        F = self._get_F(w_latent_learn, w_latent_fix)
        F_exp = F.expand(qr2_samples.shape[0], -1, -1, -1)

        # fi = F0(p)
        # Use grid_sample to extract the features for p
        f_i = torch.nn.functional.grid_sample(
            F0_exp.float(), p_init_samples.float(), mode="bilinear", align_corners=False
        )

        # F(q)
        # Use grid_sample to extract the features for q, where q € neighbours(p, r2)
        Fq = torch.nn.functional.grid_sample(F_exp.float(), qr2_samples.float(), mode="bilinear", align_corners=False)

        # Compute pairwise distances between fi and F(q).
        distances = cdist_p_norm(f_i, Fq, p=distance_l_type)

        # Get the minimum distance for each p
        min_distance_indices = torch.argmin(distances, dim=-1)

        # Translate the index to the original feature coordinates
        qr2_samples_min = torch.gather(
            qr2_samples.reshape(Fq.shape[0], -1, 2),
            dim=1,
            index=min_distance_indices[..., None].expand(-1, -1, 2),
        )

        # Define the new p
        new_p = qr2_samples_min[:, 0]

        if pbar is not None:
            info_dict = pbar.info_dict.copy()
            info_dict["distance(p, t)"] = torch.norm(t - new_p).tolist()
            pbar.set_infos(info_dict)
            pbar.update(n=0)

        return new_p

    def generate_image_from_split_w_latent(self, w_latent_learn: torch.Tensor, w_latent_fix: torch.Tensor) -> Image:
        w_latent = torch.cat((w_latent_learn, w_latent_fix), dim=1)
        return self.generate(w_latent)

    def draw_p_image(self, img_pil: Image, p: torch.Tensor, t: torch.Tensor):
        return draw_p_image(img_pil, p, t, self._input_size)

    def generate(self, w_latent: torch.Tensor) -> Image:
        with torch.no_grad():
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                img = self._G.synthesis(w_latent, noise_mode="const")
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img = Image.fromarray(img[0].cpu().numpy(), "RGB")

        return img

    def project(self, img: Image, *args, **kargs) -> torch.Tensor:
        # Prepare image for projection
        target_pil = img.convert("RGB")
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((self._G.img_resolution, self._G.img_resolution), Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        target = torch.tensor(target_uint8.transpose([2, 0, 1]), device=self._device)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            projected_w_steps = project(
                G=self._G,
                target=target,
                device=self._device,
                *args,
                **kargs,
            )
        projected_w = projected_w_steps[-1].unsqueeze(0)

        return projected_w

    def init(
        self,
        w_latent: torch.Tensor,
        trainable_w_dims: int,
        p_in_pixels: torch.Tensor,
        r1_in_pixels: torch.Tensor,
        r2_in_pixels: torch.Tensor,
        t_in_pixels: torch.Tensor,
        magnitude_direction_in_pixels: float = 1,
        mask_in_pixels: Optional[torch.Tensor] = None,
        motion_lr: float = 2e-3,
        optimizer: Optional[optim.Optimizer] = None,
    ):
        # Define dimensions of w_latent that will be learned
        w_latent_learn = w_latent[:, :trainable_w_dims]
        w_latent_learn.requires_grad = True
        w_latent_fix = w_latent[:, trainable_w_dims:]
        w_latent_fix.requires_grad = False

        p_in_pixels = p_in_pixels.to(self._device)
        r1_in_pixels = r1_in_pixels.to(self._device)
        r2_in_pixels = r2_in_pixels.to(self._device)
        t_in_pixels = t_in_pixels.to(self._device)

        # Convert into normalized -1 to 1 coordinates
        p = fix_point(p_in_pixels, self._input_size)
        r1 = fix_radius(r1_in_pixels, self._input_size)
        r2 = fix_radius(r2_in_pixels, self._input_size)
        t = fix_point(t_in_pixels, self._input_size)
        magnitude_direction = fix_radius(magnitude_direction_in_pixels, self._input_size)

        # Create the preservation mask
        if mask_in_pixels is not None:
            mask_in_pixels = mask_in_pixels.to(self._device)
            M = generate_motion_masks(
                mask_in_pixels,
                output_size=self._features_extractor_size,
                dims=self._features_extractor_dims,
            ).to(self._device)

        else:
            M = None

        if optimizer is None:
            optimizer = optim.AdamW([w_latent_learn], lr=motion_lr)

        p_init = p.detach().clone()
        F0 = self._get_F(w_latent_learn, w_latent_fix).detach().clone()

        return (
            w_latent_learn,
            w_latent_fix,
            p,
            r1,
            r2,
            t,
            magnitude_direction,
            M,
            optimizer,
            p_init,
            F0,
        )

    def step(
        self,
        optimizer: optim.Optimizer,
        motion_lambda: float,
        w_latent_learn: torch.Tensor,
        w_latent_fix: torch.Tensor,
        F0: torch.Tensor,
        p_init: torch.Tensor,
        p: torch.Tensor,
        t: torch.Tensor,
        r1: torch.Tensor,
        r1_interpolation_samples: int,
        r2: torch.Tensor,
        r2_interpolation_samples: int,
        M: torch.Tensor,
        magnitude_direction: float,
        motion_steps: int = 1,
        pbar: Optional[qqdm] = None,
    ):
        self._compute_motion_step(
            F0=F0,
            p=p,
            t=t,
            r1=r1,
            M=M,
            w_latent_learn=w_latent_learn,
            w_latent_fix=w_latent_fix,
            optimizer=optimizer,
            steps=motion_steps,
            lambda_value=motion_lambda,
            magnitude_direction=magnitude_direction,
            r1_interpolation_samples=r1_interpolation_samples,
            pbar=pbar,
        )

        # Point tracking step
        p = self._compute_point_tracking(
            F0=F0,
            w_latent_learn=w_latent_learn,
            w_latent_fix=w_latent_fix,
            p=p,
            p_init=p_init,
            r2=r2,
            t=t,
            r2_interpolation_samples=r2_interpolation_samples,
            pbar=pbar,
        )

        return p

    def compute(
        self,
        w_latent: torch.Tensor,
        trainable_w_dims: int,
        p_in_pixels: torch.Tensor,
        r1_in_pixels: torch.Tensor,
        r2_in_pixels: torch.Tensor,
        t_in_pixels: torch.Tensor,
        magnitude_direction_in_pixels: float = 1,
        r1_interpolation_samples: int = 3,
        r2_interpolation_samples: int = 12,
        mask_in_pixels: Optional[torch.Tensor] = None,
        steps: int = 300,
        motion_steps: int = 1,
        motion_lambda: float = 20.0,
        motion_lr: float = 2e-3,
        debug_folder_path: Path = Path(__file__).parent / "debug",
        debug_draw_original_image: bool = True,
        debug_draw_step_image: Union[int, bool] = 5,
        return_image: bool = True,
    ) -> torch.Tensor:
        (
            w_latent_learn,
            w_latent_fix,
            p,
            r1,
            r2,
            t,
            magnitude_direction,
            M,
            optimizer,
            p_init,
            F0,
        ) = self.init(
            w_latent=w_latent,
            trainable_w_dims=trainable_w_dims,
            p_in_pixels=p_in_pixels,
            r1_in_pixels=r1_in_pixels,
            r2_in_pixels=r2_in_pixels,
            t_in_pixels=t_in_pixels,
            magnitude_direction_in_pixels=magnitude_direction_in_pixels,
            mask_in_pixels=mask_in_pixels,
            motion_lr=motion_lr,
        )

        # Create image
        if debug_draw_original_image:
            debug_folder_path.mkdir(parents=True, exist_ok=True)

            img_orig_pil = self.generate_image_from_split_w_latent(w_latent_learn, w_latent_fix)
            img_orig_pil = draw_p_image(img_orig_pil, p, t, self._input_size)
            img_orig_pil.save(debug_folder_path / "init.png")

        pbar = qqdm(total=steps, desc="Computing DragGAN")
        for global_step in range(steps):
            p = self.step(
                optimizer=optimizer,
                motion_lambda=motion_lambda,
                w_latent_learn=w_latent_learn,
                w_latent_fix=w_latent_fix,
                F0=F0,
                p_init=p_init,
                p=p,
                t=t,
                r1=r1,
                r1_interpolation_samples=r1_interpolation_samples,
                r2=r2,
                r2_interpolation_samples=r2_interpolation_samples,
                M=M,
                magnitude_direction=magnitude_direction,
                motion_steps=motion_steps,
                pbar=pbar,
            )

            if not isinstance(debug_draw_step_image, bool) and global_step % debug_draw_step_image == 0:
                debug_folder_path.mkdir(parents=True, exist_ok=True)

                img_step_pil = self.generate_image_from_split_w_latent(w_latent_learn, w_latent_fix)
                img_step_pil = draw_p_image(img_step_pil, p, t, self._input_size)
                img_step_pil.save(debug_folder_path / f"step_{global_step:04d}.png")

            pbar.update(n=1)

        # Return the results
        if return_image:
            img_final_pil = self.generate_image_from_split_w_latent(w_latent_learn, w_latent_fix)
            return w_latent, img_final_pil

        return w_latent
