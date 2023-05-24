import cv2
from PIL import Image
import torch
from torch import nn
from torch import optim
import numpy as np
from qqdm import qqdm
from typing import Optional, Iterable, Literal, Union
from functools import partial
from torch.nn import functional as Functional

import sys
from pathlib import Path
CURR_PATH = Path(__file__).parent
sys.path.append(str((CURR_PATH / "stylegan2-ada-pytorch").absolute()))
import dnnlib
import legacy

def fix_point(
    p: torch.Tensor,
    input_size: int
) -> torch.Tensor:
    scale_factor = 2. / input_size
    p_fix = p * scale_factor - 1
    return p_fix

def fix_radius(
    r: torch.Tensor,
    input_size: int
) -> torch.Tensor:
    scale_factor = 2. / input_size
    r_fix = r * scale_factor
    return r_fix

def fix_magnitude_direction(
    magnitude_direction: float,
    input_size: int
) -> float:
    scale_factor = 2. / input_size

    magnitude_direction_fix = magnitude_direction * scale_factor

    return magnitude_direction_fix

def generate_motion_masks(
    p: torch.Tensor, 
    r: torch.Tensor, 
    resolution: int = 5, 
) -> torch.Tensor:
    p_aux = p.reshape(p.shape[0], 1, 1, 2)
    q_mask = p_aux.expand(-1, resolution, resolution, -1) # N x step x step x 2
    
    # Inject the radius
    radius_meshgrid = torch.stack([
        torch.stack(torch.meshgrid(2 * [torch.linspace(-r_i, r_i, resolution)]), dim=-1)
        for r_i in r
    ], dim=0).to(q_mask.device) # N x step x step x 2

    q_mask = torch.clip(q_mask + radius_meshgrid, -1.0, 1.0)        

    return q_mask

def generate_motion_direction(p: torch.Tensor, t: torch.Tensor, magnitude_direction: float = 0.1) -> torch.Tensor:
    dir_pt = magnitude_direction * Functional.normalize(t - p, dim=-1)
    return dir_pt

def generate_reg_masks(M, output_size=256, dims=128) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        M.reshape(1, 1, M.shape[0], M.shape[1]).float(), 
        size=[output_size, output_size], mode='nearest') \
        .reshape(1, 1, output_size, output_size) \
        .expand(-1, dims, -1, -1)

def get_F(w_latent_learn: torch.Tensor, w_latent_fix: torch.Tensor, G: nn.Module) -> torch.Tensor:
    F_r = [None]
    def forward_layer_hook(F_r, module, input, output):
        F_r[0] = output[0]

    G.synthesis.b256.register_forward_hook(partial(forward_layer_hook, F_r))

    # Features
    w_latent = torch.cat((w_latent_learn, w_latent_fix), dim=1)
    _ = G.synthesis(w_latent, noise_mode='const')
    if w_latent.device.type == "cuda":
        torch.cuda.synchronize()

    return F_r[0]

def cdist_p_norm(x: torch.Tensor, y: torch.Tensor, p: int = 1) -> torch.Tensor:
    x = x[:, :, None].expand(-1, -1, y.shape[1], -1)
    y = y[:, None].expand(-1, x.shape[1], -1, -1)
    return torch.norm(x - y, p=p, dim=-1)

def compute_motion_step(
    pbar: qqdm, 
    G: nn.Module,
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
    q_size: int = 3,
    magnitude_direction: int = .1
):
    # Generate masks
    qr1_mask = generate_motion_masks(p, r1, resolution=q_size)
    qr1d_mask = qr1_mask + generate_motion_direction(p, t, magnitude_direction=magnitude_direction)

    # Start training
    for step_idx in range(steps):
        optimizer.zero_grad()
    
        F = get_F(w_latent_learn, w_latent_fix, G)
        # print(w_latent_learn.std((0, 2)))
        # print(w_latent_learn.std((0, 2)))
        # print(w_latent_learn.std((0, 2)))
        F_exp = F.expand(qr1_mask.shape[0], -1, -1, -1) # N x C x H x W

        # F(q)
        # aux = torch.stack(torch.meshgrid((torch.arange(0, 256), torch.arange(0, 256))), dim=0)[None, ...]
        # aux2 = torch.nn.functional.grid_sample(aux.float(), torch.tensor([1., 1.]).reshape(1, 1, 1, 2), mode='bilinear', align_corners=False)
        # print(aux.shape, aux2)
        # print(aux.shape, aux2)
        # exit()
        Fq = torch.nn.functional.grid_sample(F_exp.float(), qr1_mask.float(), mode='bilinear', align_corners=False).detach()

        # F(q + d)
        Fqd = torch.nn.functional.grid_sample(F_exp.float(), qr1d_mask.float(), mode='bilinear', align_corners=False)

        motion_loss = l1_loss(Fq, Fqd)
        # reg_loss = l1_loss(F * (1 - M), F0 * (1 - M))

        loss = motion_loss  # + lambda_value * reg_loss

        info_dict = pbar.info_dict.copy()
        info_dict['loss'] = float(loss)
        info_dict['motion_step'] = f"{step_idx}/{steps}"
        pbar.set_infos(info_dict)
        
        loss.backward()
        optimizer.step()
        pbar.update(n=0)

@torch.no_grad()
def compute_point_tracking(
    pbar: qqdm,
    G: nn.Module,
    F0: torch.Tensor,
    w_latent_learn: torch.Tensor,
    w_latent_fix: torch.Tensor,
    p: torch.Tensor,
    p_init: torch.Tensor,
    r2: torch.Tensor,
    resolution: int,
    distance_type: Literal["l1", "l2"] = "l1"
):
    p_init_mask = p_init.reshape(p_init.shape[0], 1, 1, 2)
    qr2_mask = generate_motion_masks(p, r=r2, resolution=resolution)
    
    F0_exp = F0.expand(qr2_mask.shape[0], -1, -1, -1)    
    F = get_F(w_latent_learn, w_latent_fix, G)
    F_exp = F.expand(qr2_mask.shape[0], -1, -1, -1) # N x C x H x W

    f_i = torch.nn.functional.grid_sample(F0_exp.float(), p_init_mask.float(), mode='bilinear', align_corners=False)
    Fq = torch.nn.functional.grid_sample(F_exp.float(), qr2_mask.float(), mode='bilinear', align_corners=False)
    
    f_i_mat = f_i.reshape(f_i.shape[0], f_i.shape[1], -1).permute(0, 2, 1) # B x L x C
    Fq_mat = Fq.reshape(Fq.shape[0], Fq.shape[1], -1).permute(0, 2, 1) # B x P x C

    if distance_type == "l1":
        distances = cdist_p_norm(f_i_mat, Fq_mat, p=1)

    elif distance_type == "l2":
        distances = torch.cdist(
            f_i_mat,
            Fq_mat
        )
        
    else:
        raise ValueError("Wrong distance_type")


    min_distance_indices = torch.argmin(distances, dim=-1)
    qr2_mask_min = torch.gather(
        qr2_mask.reshape(Fq.shape[0], -1, 2), 
        dim=1, 
        index=min_distance_indices[..., None].expand(-1, -1, 2)
    )

    new_p = qr2_mask_min[0].clone()
    # print(q_mask, t)
    # print(q_mask, t)
    # print(q_mask, t)
    # print(q_mask, t)
    # print(new_p, t)
    # print(new_p, t)
    # print(new_p, t)
    # print(new_p, t)
    
    info_dict = pbar.info_dict.copy()
    info_dict['distance(p, t)'] = torch.norm(t - new_p).tolist()
    pbar.set_infos(info_dict)
    pbar.update(n=0)

    return new_p

def l1_loss(x, y):
    return abs(x - y).mean()

def generate_image(G: nn.Module, w_latent_learn: torch.Tensor, w_latent_fix: torch.Tensor):
    with torch.no_grad():
        w_latent = torch.cat((w_latent_learn, w_latent_fix), dim=1)
        for idx, w in enumerate(w_latent):
            img = G.synthesis(w.unsqueeze(0), noise_mode='const')
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
            
    return img
        
def draw_p_image(G: nn.Module, w_latent_learn: torch.Tensor, w_latent_fix: torch.Tensor, p: torch.Tensor, t: torch.Tensor, input_size: int):
    img_pil = generate_image(G, w_latent_learn, w_latent_fix)
    img_cv = np.array(img_pil)
    for p_i, t_i in zip(p, t):
        p_i_pixels = (p_i + 1) / 2. * input_size
        t_i_pixels = (t_i + 1) / 2. * input_size

        rad_draw = int(input_size * 0.02)
        cv2.circle(img_cv, (int(p_i_pixels[0]), int(p_i_pixels[1])), rad_draw, (255, 0, 0), -1)
        cv2.circle(img_cv, (int(t_i_pixels[0]), int(t_i_pixels[1])), rad_draw, (0, 0, 255), -1)
    return Image.fromarray(img_cv)
    

if __name__ == '__main__':
    seed = 42
    device = 'cuda:0'
    network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl"
    results_folder_path = CURR_PATH / "results"
    truncation_psi = 0.85
    draw_interval = 5
    global_steps = 300
    motion_steps = 1
    motion_lambda = 20
    motion_lr = 2e-3
    num_first_latents = 6
    q_size = 3
    p_pixels = torch.tensor([(320, 316)]).to(device)
    r1_pixels = torch.tensor([3]).to(device)
    r2_pixels = torch.tensor([12]).to(device)
    # t_pixels = torch.tensor([(420, 243)]).to(device)
    t_pixels = torch.tensor([(220, 243)]).to(device)
    magnitude_direction_pixels = 1

    # Create folder for results
    results_folder_path.mkdir(exist_ok=True, parents=True)

    # Load model ckpt
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    input_size = G.synthesis.img_resolution

    # Compute first mapping
    with torch.no_grad():
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        c = torch.zeros([1, G.c_dim], device=device)

        w_latent_orig = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=None).detach().clone()
        w_latent_learn, w_latent_fix = torch.split(
            w_latent_orig, 
            [num_first_latents, w_latent_orig.shape[1] - num_first_latents], 
            dim=1
        )

    w_latent_learn.requires_grad = True
    w_latent = torch.cat((w_latent_learn, w_latent_fix), dim=1)

    # Convert into normalized -1 to 1 coordinates
    p = fix_point(p_pixels, input_size)
    r1 = fix_radius(r1_pixels, input_size)
    r2 = fix_radius(r2_pixels, input_size)
    t = fix_point(t_pixels, input_size) 
    magnitude_direction = fix_magnitude_direction(magnitude_direction_pixels, input_size)

    # Create image
    img_orig_pil = draw_p_image(G, w_latent_learn, w_latent_fix, p, t, input_size)
    img_orig_pil.save(results_folder_path / "groundtruth.png")
   

    # for i in range(q_mask.shape[1]):
    #     for j in range(q_mask.shape[2]):
    #         print(f"[{i} - {j}] {q_mask[0, i, j, 0]}, {q_mask[0, i, j, 1]} <-> {qd_mask[0, i, j, 0]}, {qd_mask[0, i, j, 1]}")
    # exit()
    
    Maux = torch.ones((input_size, input_size), dtype=torch.bool)
    # Maux[t_pixels[0, 0] - 75:t_pixels[0, 0] + 75, t_pixels[0, 1] - 75:t_pixels[0, 1] + 75] = 1
    M = generate_reg_masks(Maux, output_size=256, dims=3).to(device)
    
    pbar = qqdm(total=global_steps, desc="Computing DragGAN")
    optimizer = optim.AdamW([w_latent_learn], lr=motion_lr) 

    p_init = p.detach().clone()
    F0 = get_F(w_latent_learn, w_latent_fix, G).detach().clone()
    for global_step in range(global_steps):
        # Motion supervision step
        Fq = compute_motion_step(
            pbar,
            G,
            F0,
            p,
            t,
            r1,
            M, 
            w_latent_learn,
            w_latent_fix, 
            optimizer,
            motion_steps,
            motion_lambda,
            q_size,
            magnitude_direction
        )
        
    
        # Point tracking step
        p = compute_point_tracking(
            pbar,
            G,
            F0,
            w_latent_learn,
            w_latent_fix,
            p,
            p_init,
            r2,
            q_size
        )

        if global_step % draw_interval == 0:
            img_orig_pil = draw_p_image(G, w_latent_learn, w_latent_fix, p, t, input_size)
            img_orig_pil.save(results_folder_path / f"pred_{global_step:04d}.png")

        pbar.update(n=1)
        

    # Create image
    img_last_draw = draw_p_image(G, w_latent_learn, w_latent_fix, p, t, input_size)
    img_last_draw.save(results_folder_path / "pred_final.png")

    img_gen_pil = generate_image(G, w_latent_learn, w_latent_fix)
    img_gen_pil.save(results_folder_path / "result.png")