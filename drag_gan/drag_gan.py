from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from qqdm import qqdm
from torch import optim
from torch.nn import functional as Functional

from drag_gan.generators import REGISTERED_GENERATORS, BaseGenerator
from drag_gan.utils import TrainableLatent


# INTERNAL FUNCTIONS
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


def generate_motion_masks(mask_in_pixels, output_size: int = 256) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        mask_in_pixels.reshape(1, 1, mask_in_pixels.shape[0], mask_in_pixels.shape[1]).float(),
        size=[output_size, output_size],
        mode="nearest",
    ).reshape(1, 1, output_size, output_size)


def draw_image_with_points(image_pil: Image, p: torch.Tensor, t: torch.Tensor, input_size: int):
    image_cv = np.array(image_pil)
    for p_i, t_i in zip(p, t):
        p_i_pixels = (p_i + 1) / 2.0 * input_size
        t_i_pixels = (t_i + 1) / 2.0 * input_size

        rad_draw = int(input_size * 0.02)
        cv2.circle(image_cv, (int(p_i_pixels[0]), int(p_i_pixels[1])), rad_draw, (255, 0, 0), -1)
        cv2.circle(image_cv, (int(t_i_pixels[0]), int(t_i_pixels[1])), rad_draw, (0, 0, 255), -1)
    return Image.fromarray(image_cv)


class DragGAN:
    REGISTERED_GENERATORS = REGISTERED_GENERATORS

    def __init__(
        self,
        generator: Optional[BaseGenerator] = None,
    ):
        self.generator = generator

    # UTILS
    @property
    def params(self) -> dict[str, Any]:
        assert self.generator is not None
        return self.generator.params

    def pixel_coord_to_norm_coord(self, p: torch.Tensor):
        assert self.generator is not None
        return self.pixel_value_to_norm_value(p) - 1

    def norm_coord_to_pixel_coord(self, p: torch.Tensor):
        assert self.generator is not None
        return self.norm_value_to_pixel_value(p + 1)

    def pixel_value_to_norm_value(self, r: torch.Tensor):
        assert self.generator is not None
        scale_factor = 2.0 / self.generator.input_size
        r_fix = r * scale_factor
        return r_fix

    def norm_value_to_pixel_value(self, r: torch.Tensor):
        assert self.generator is not None
        scale_factor = self.generator.input_size / 2.0
        r_fix = r * scale_factor
        return r_fix

    # Generate and project
    def get_latent_from_seed(self, seed: int, *args, **kargs):
        assert self.generator is not None
        return self.generator.get_latent_from_seed(seed, *args, **kargs)

    def generate(self, latent: Union[TrainableLatent, torch.Tensor]) -> Image:
        assert self.generator is not None
        if isinstance(latent, TrainableLatent):
            return self.generator.generate(latent.latent)

        else:
            assert isinstance(latent, torch.Tensor)
            return self.generator.generate(latent)

    def project(self, image: Image, *args: list[Any], **kargs: dict[str, Any]) -> TrainableLatent:
        assert self.generator is not None
        return self.generator.project(image, *args, **kargs)

    # DragGAN init, step and compute
    def init(
        self,
        trainable_latent: TrainableLatent,
        p_in_pixels: torch.Tensor,
        r1_in_pixels: torch.Tensor,
        r2_in_pixels: torch.Tensor,
        t_in_pixels: torch.Tensor,
        magnitude_direction_in_pixels: float = 1,
        mask_in_pixels: Optional[torch.Tensor] = None,
        motion_lr: float = 2e-3,
        optimizer: Optional[optim.Optimizer] = None,
    ):
        assert self.generator is not None
        p_in_pixels = p_in_pixels.to(self.generator.device)
        r1_in_pixels = r1_in_pixels.to(self.generator.device)
        r2_in_pixels = r2_in_pixels.to(self.generator.device)
        t_in_pixels = t_in_pixels.to(self.generator.device)

        # Convert into normalized -1 to 1 coordinates
        p = self.pixel_coord_to_norm_coord(p_in_pixels)
        r1 = self.pixel_value_to_norm_value(r1_in_pixels)
        r2 = self.pixel_value_to_norm_value(r2_in_pixels)
        t = self.pixel_coord_to_norm_coord(t_in_pixels)
        magnitude_direction = self.pixel_value_to_norm_value(magnitude_direction_in_pixels)

        # Create the preservation mask
        if mask_in_pixels is not None:
            mask_in_pixels = mask_in_pixels.to(self.generator.device)
            M = generate_motion_masks(
                mask_in_pixels,
                output_size=self.generator.features_extractor_size,
            ).to(self.generator.device)

        else:
            M = None

        if optimizer is None:
            optimizer = optim.AdamW([trainable_latent.trainable_latent], lr=motion_lr)

        p_init = p.detach().clone()
        F0 = self.generator.features(trainable_latent.latent).detach().clone()

        return (
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
        trainable_latent: TrainableLatent,
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
            trainable_latent=trainable_latent,
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
            trainable_latent=trainable_latent,
            p=p,
            p_init=p_init,
            r2=r2,
            t=t,
            r2_interpolation_samples=r2_interpolation_samples,
            pbar=pbar,
        )

        return p

    def _compute_motion_step(
        self,
        F0: torch.Tensor,
        p: torch.Tensor,
        t: torch.Tensor,
        r1: torch.Tensor,
        M: torch.Tensor,
        trainable_latent: TrainableLatent,
        optimizer: optim,
        steps: int = 300,
        lambda_value: float = 12.0,
        magnitude_direction: int = 0.1,
        r1_interpolation_samples: int = 3,
        pbar: Optional[qqdm] = None,
    ):
        assert self.generator is not None

        # Generate masks
        qr1_samples = generate_motion_samples(p, r1, samples=r1_interpolation_samples)
        qr1d_samples = qr1_samples + generate_motion_direction(p, t, magnitude_direction=magnitude_direction)

        # Start training
        for step_idx in range(steps):
            optimizer.zero_grad()

            # Extract the features from the current w_latent
            F = self.generator.features(trainable_latent.latent)
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
        trainable_latent: TrainableLatent,
        p: torch.Tensor,
        p_init: torch.Tensor,
        r2: torch.Tensor,
        t: torch.Tensor,
        r2_interpolation_samples: int = 12,
        distance_l_type: int = 1,
        pbar: Optional[qqdm] = None,
    ):
        assert self.generator is not None

        # Generate p an q € neighbours(p, r2)
        p_init_samples = p_init.reshape(p_init.shape[0], 1, 1, 2)
        qr2_samples = generate_motion_samples(p, r=r2, samples=r2_interpolation_samples)

        # Extract the features from the current w_latent
        F0_exp = F0.expand(qr2_samples.shape[0], -1, -1, -1)
        F = self.generator.features(trainable_latent.latent)
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

    def compute(
        self,
        trainable_latent: TrainableLatent,
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
        assert self.generator is not None

        (
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
            trainable_latent=trainable_latent,
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

            image_orig_pil = self.generator.generate(trainable_latent.latent)
            image_orig_pil = draw_image_with_points(image_orig_pil, p, t, self.generator.input_size)
            image_orig_pil.save(debug_folder_path / "init.png")

        pbar = qqdm(total=steps, desc="Computing DragGAN")
        for global_step in range(steps):
            p = self.step(
                optimizer=optimizer,
                motion_lambda=motion_lambda,
                trainable_latent=trainable_latent,
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

                image_step_pil = self.generator.generate(trainable_latent.latent)
                image_step_pil = draw_image_with_points(image_step_pil, p, t, self.generator.input_size)
                image_step_pil.save(debug_folder_path / f"step_{global_step:04d}.png")

            pbar.update(n=1)

        # Return the results
        if return_image:
            image_final_pil = self.generator.generate(trainable_latent.latent)
            return trainable_latent.latent, image_final_pil

        return trainable_latent.latent
