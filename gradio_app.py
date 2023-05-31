from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from drag_gan.drag_gan import DragGAN
from drag_gan.generators import BaseGenerator, StyleGANv2Generator
from drag_gan.utils import on_change_single_global_state


def set_generator_parameters(global_state):
    global_state["model"].generator.params = global_state["generator_params"]


def get_model(global_state):
    drag_gan: DragGAN = global_state["model"]
    set_generator_parameters(global_state)
    return drag_gan


def generate_from_seed(drag_gan: DragGAN, seed: int):
    trainable_latent = drag_gan.get_latent_from_seed(int(seed))
    image_raw = drag_gan.generate(trainable_latent)
    return trainable_latent, image_raw


def init_drag_gan(generator: BaseGenerator, seed: int):
    drag_gan = DragGAN(generator)
    trainable_latent, image_orig = generate_from_seed(drag_gan, seed)
    return drag_gan, trainable_latent, image_orig


def init_drag_gan_from_path_or_url(path_or_url, global_state, seed, model_value):
    drag_gan: DragGAN = global_state["model"]
    drag_gan.generator = DragGAN.REGISTERED_GENERATORS[model_value].load_from_path(path_or_url)
    trainable_latent, image_raw = generate_from_seed(drag_gan, seed)

    create_images(image_raw, global_state)

    global_state["temporal_params"] = {}

    global_state["model"] = drag_gan
    set_generator_parameters(global_state)
    global_state["temporal_params"]["trainable_latent"] = trainable_latent

    # Restart draw
    return global_state, image_raw, global_state["draws"]["image_with_mask"]


font = ImageFont.truetype(str(Path(__file__).parent / "misc/Roboto-Medium.ttf"), 32)


def draw_points_on_image(image, points, curr_point=None):
    overlay_rgba = Image.new("RGBA", image.size, 0)
    overlay_draw = ImageDraw.Draw(overlay_rgba)
    for point_key, point in points.items():
        if curr_point is not None and curr_point == point_key:
            p_color = (255, 0, 0)
            t_color = (0, 0, 255)

        else:
            p_color = (255, 0, 0, 35)
            t_color = (0, 0, 255, 35)

        rad_draw = int(image.size[0] * 0.02)

        p_start = point.get("start_temp", point["start"])
        p_target = point["target"]

        if p_start is not None and p_target is not None:
            p_draw = int(p_start[0]), int(p_start[1])
            t_draw = int(p_target[0]), int(p_target[1])

            overlay_draw.line(
                (p_draw[0], p_draw[1], t_draw[0], t_draw[1]),
                fill=(255, 255, 0),
                width=2,
            )

        if p_start is not None:
            p_draw = int(p_start[0]), int(p_start[1])
            overlay_draw.ellipse(
                (
                    p_draw[0] - rad_draw,
                    p_draw[1] - rad_draw,
                    p_draw[0] + rad_draw,
                    p_draw[1] + rad_draw,
                ),
                fill=p_color,
            )

            if curr_point is not None and curr_point == point_key:
                overlay_draw.text(p_draw, "p", font=font, align="center", fill=(0, 0, 0))

        if p_target is not None:
            t_draw = int(p_target[0]), int(p_target[1])
            overlay_draw.ellipse(
                (
                    t_draw[0] - rad_draw,
                    t_draw[1] - rad_draw,
                    t_draw[0] + rad_draw,
                    t_draw[1] + rad_draw,
                ),
                fill=t_color,
            )

            if curr_point is not None and curr_point == point_key:
                overlay_draw.text(t_draw, "t", font=font, align="center", fill=(0, 0, 0))

    return Image.alpha_composite(image.convert("RGBA"), overlay_rgba).convert("RGB")


def draw_mask_on_image(image, mask):
    im_mask = np.uint8(mask * 255)
    im_mask_rgba = np.concatenate(
        (
            np.tile(im_mask[..., None], [1, 1, 3]),
            45 * np.ones((im_mask.shape[0], im_mask.shape[1], 1), dtype=np.uint8),
        ),
        axis=-1,
    )
    im_mask_rgba = Image.fromarray(im_mask_rgba).convert("RGBA")

    return Image.alpha_composite(image.convert("RGBA"), im_mask_rgba).convert("RGB")


def create_images(image_raw, global_state):
    global_state["images"]["image_orig"] = image_raw.copy()
    global_state["images"]["image_raw"] = image_raw
    global_state["draws"]["image_with_points"] = draw_points_on_image(
        image_raw, global_state["points"], global_state["curr_point"]
    )

    global_state["images"]["image_mask"] = np.ones((image_raw.size[1], image_raw.size[0]), dtype=np.uint8)
    global_state["draws"]["image_with_mask"] = draw_mask_on_image(
        global_state["images"]["image_raw"], global_state["images"]["image_mask"]
    )


def main(
    generator: BaseGenerator,
    default_seed: int = 42,
    device: str = "cuda:0",
):
    css = """
    .image_nonselectable img {
        -webkit-user-drag: none;
        -ms-user-drag: none;
        -moz-user-drag: none;
        -o-user-drag: none;
        user-drag: none;
    }
    """

    tutorial = """
### Pair-points
1. Create a new pair-point using the "Pair-Points" tab and the "Add point" button.
2. Once clicked, check the "List of pair-points", a new one will appear.
3. Seleccione en Type of point si desea crear "start (p)" o el "target (t)".
4. Select in Type of point if you want to create "start (p)" or the "target (t)".
5. Click on the image to create this point.
6. You can select in the "List of pair-points" and you can modify it.
7. You can remove a pair of points by clicking on Remove pair-point.

### Mask
1. Select the "Mask" tab.
2. Once selected, you will be able to draw subtractively. The default mask is the entire image that can be edited.
3. By drawing you can indicate areas that you want not to be edited.

### Run
1. Click on "Start".
2. Click on "Stop" and make the manual changes of the control pair-points that you want.
    """

    about_paper = """
### Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold
> Xingang Pan, Ayush Tewari, Thomas Leimkühler, Lingjie Liu, Abhimitra Meka, Christian Theobalt<br>
> *SIGGRAPH 2023 Conference Proceedings*

Synthesizing visual content that meets users' needs often requires flexible and precise controllability of the pose, shape, expression, and layout of the generated objects. Existing approaches gain controllability of generative adversarial networks (GANs) via manually annotated training data or a prior 3D model, which often lack flexibility, precision, and generality. In this work, we study a powerful yet much less explored way of controlling GANs, that is, to "drag" any points of the image to precisely reach target points in a user-interactive manner, as shown in Fig.1. To achieve this, we propose DragGAN, which consists of two main components: 1) a feature-based motion supervision that drives the handle point to move towards the target position, and 2) a new point tracking approach that leverages the discriminative generator features to keep localizing the position of the handle points. Through DragGAN, anyone can deform an image with precise control over where pixels go, thus manipulating the pose, shape, expression, and layout of diverse categories such as animals, cars, humans, landscapes, etc. As these manipulations are performed on the learned generative image manifold of a GAN, they tend to produce realistic outputs even for challenging scenarios such as hallucinating occluded content and deforming shapes that consistently follow the object's rigidity. Both qualitative and quantitative comparisons demonstrate the advantage of DragGAN over prior approaches in the tasks of image manipulation and point tracking. We also showcase the manipulation of real images through GAN inversion.
    """

    with gr.Blocks(css=css) as app:
        drag_gan, trainable_latent, image_orig = init_drag_gan(
            StyleGANv2Generator.load_from_pretrained("afhqwild"), 42
        )  # The force of the lion!

        global_state = gr.State(
            {
                "images": {
                    # image_orig
                    # image_raw
                    # image_mask
                },
                "draws": {
                    # image_with_points
                    # image_with_mask
                },
                "temporal_params": {
                    "trainable_latent": trainable_latent,
                },
                "generator_params": {},
                "params": {
                    "motion_lr": 2e-3,
                    "motion_lambda": 20,
                    "r1_in_pixels": 3,
                    "r2_in_pixels": 12,
                    "magnitude_direction_in_pixels": 1.0,
                },
                "device": device,
                "draw_interval": 5,
                "radius_mask": 51,
                "model": drag_gan,
                "points": {},
                "curr_point": None,
                "curr_type_point": "start",
            }
        )
        create_images(image_orig, global_state.value)

        with gr.Row():
            # Left column
            with gr.Column(scale=0.7):
                with gr.Accordion("Information"):
                    with gr.Tab("Tutorial"):
                        gr.Markdown(tutorial)
                    with gr.Tab("About the paper"):
                        gr.Markdown(about_paper)

                with gr.Accordion("Network & latent"):
                    with gr.Row():
                        form_model_dropdown = gr.Dropdown(
                            choices=list(DragGAN.REGISTERED_GENERATORS.keys()),
                            label="Models",
                            value="StyleGANv2Generator",
                        )

                    with gr.Row():
                        with gr.Tab("Pretrained models"):
                            form_pretrained_dropdown = gr.Dropdown(
                                choices=list(StyleGANv2Generator.PRETRAINED_MODELS.keys()),
                                label="Pretrained model",
                                value="afhqwild",
                            )

                        with gr.Tab("Local file"):
                            form_model_pickle_file = gr.File(label="Pickle file")

                        with gr.Tab("URL"):
                            with gr.Row():
                                form_model_url = gr.Textbox(
                                    placeholder="Url of the pickle file",
                                    label="URL",
                                )
                                form_model_url_btn = gr.Button("Submit")

                    with gr.Row().style(equal_height=True):
                        with gr.Tab("Image seed"):
                            with gr.Row():
                                form_seed_number = gr.Number(
                                    value=default_seed,
                                    interactive=True,
                                    label="Seed",
                                )
                                form_update_image_seed_btn = gr.Button("Update image")

                        with gr.Tab("Image projection"):
                            with gr.Row():
                                form_project_file = gr.File(label="Image project file")
                                form_project_iterations_number = gr.Number(
                                    value=1_000,
                                    label="Image projection num steps",
                                )
                                form_update_image_project_btn = gr.Button("Run projection")

                        form_reset_image = gr.Button("Reset image")

                    with gr.Row():
                        with gr.Tab("Generator Parameters"):
                            generator.get_gradio_panel(global_state)

                with gr.Accordion("Tools"):
                    with gr.Tab("Pair-Points") as points_tab:
                        form_points_dropdown = gr.Dropdown(
                            choices=[],
                            value="",
                            interactive=True,
                            label="List of pair-points",
                        )

                        form_type_point_radio = gr.Radio(
                            ["start (p)", "target (t)"],
                            value="start (p)",
                            label="Type",
                        )

                        with gr.Row():
                            form_add_point_btn = gr.Button("Add pair-point").style(full_width=True)
                            form_remove_point_btn = gr.Button("Remove pair-point").style(full_width=True)

                    with gr.Tab("Mask (subtractive mask)") as mask_tab:
                        gr.Markdown(
                            """
                            White zone = editable by DragGAN
                            Transparent zone = not editable by DragGAN.
                        """
                        )
                        form_reset_mask_btn = gr.Button("Reset mask").style(full_width=True)
                        form_radius_mask_number = gr.Number(
                            value=global_state.value["radius_mask"],
                            interactive=True,
                            label="Radius (pixels)",
                        ).style(full_width=False)

                    with gr.Row():
                        with gr.Tab("Run"):
                            with gr.Row():
                                with gr.Column():
                                    form_start_btn = gr.Button("Start").style(full_width=True)
                                    form_stop_btn = gr.Button("Stop").style(full_width=True)
                                form_steps_number = gr.Number(value=0, label="Steps", interactive=False).style(
                                    full_width=False
                                )
                                form_draw_interval_number = gr.Number(
                                    value=global_state.value["draw_interval"],
                                    label="Draw Interval (steps)",
                                    interactive=True,
                                ).style(full_width=False)
                                form_download_result_file = gr.File(label="Download result", visible=False).style(
                                    full_width=True
                                )

                        with gr.Tab("Hyperparameters"):
                            with gr.Row():
                                form_lambda_number = gr.Number(
                                    value=global_state.value["params"]["motion_lambda"],
                                    interactive=True,
                                    label="Lambda",
                                ).style(full_width=True)
                                form_motion_lr_number = gr.Number(
                                    value=global_state.value["params"]["motion_lr"],
                                    interactive=True,
                                    label="LR",
                                ).style(full_width=True)
                                form_magnitude_direction_in_pixels_number = gr.Number(
                                    value=global_state.value["params"]["magnitude_direction_in_pixels"],
                                    interactive=True,
                                    label=("Magnitude direction of d vector" " (pixels)"),
                                ).style(full_width=True)

                            with gr.Row():
                                form_r1_in_pixels_number = gr.Number(
                                    value=global_state.value["params"]["r1_in_pixels"],
                                    interactive=True,
                                    label="R1 (pixels)",
                                ).style(full_width=False)
                                form_r2_in_pixels_number = gr.Number(
                                    value=global_state.value["params"]["r2_in_pixels"],
                                    interactive=True,
                                    label="R2 (pixels)",
                                ).style(full_width=False)

            # Right column
            with gr.Column():
                form_image_draw = gr.Image(
                    global_state.value["draws"]["image_with_points"], elem_classes="image_nonselectable"
                )
                form_image_mask_draw = gr.Image(
                    global_state.value["draws"]["image_with_mask"],
                    visible=False,
                    elem_classes="image_nonselectable",
                )
                gr.Markdown("Credits: Adrià Ciurana Lanau | info@dreamlearning.ai")

        # Network & latents tab listeners
        def on_change_model(model_value, global_state):
            model: DragGAN = get_model(global_state)
            model.generator = model.REGISTERED_GENERATORS[model_value]()

            return gr.Dropdown.update(choices=list(model.generator.PRETRAINED_MODELS.keys()))

        form_model_dropdown.change(
            on_change_model, inputs=[form_model_dropdown, global_state], outputs=[form_pretrained_dropdown]
        )

        def on_change_pretrained_dropdown(pretrained_value, global_state, seed):
            model: DragGAN = get_model(global_state)
            model.generator = model.generator.load_from_pretrained(pretrained_value)
            trainable_latent, image_raw = generate_from_seed(model, seed)

            create_images(image_raw, global_state)

            # Restart draw
            global_state["temporal_params"] = {"trainable_latent": trainable_latent}

            return global_state, image_raw, global_state["draws"]["image_with_mask"]

        form_pretrained_dropdown.change(
            on_change_pretrained_dropdown,
            inputs=[form_pretrained_dropdown, global_state, form_seed_number],
            outputs=[global_state, form_image_draw, form_image_mask_draw],
        )

        def on_change_model_pickle(model_pickle_file, global_state, seed, model_value):
            return init_drag_gan_from_path_or_url(model_pickle_file.name, global_state, seed, model_value)

        form_model_pickle_file.change(
            on_change_model_pickle,
            inputs=[form_model_pickle_file, global_state, form_seed_number, form_model_dropdown],
            outputs=[global_state, form_image_draw, form_image_mask_draw],
        )

        def on_change_model_url(url, global_state, seed, model_value):
            return init_drag_gan_from_path_or_url(url, global_state, seed, model_value)

        form_model_url_btn.click(
            on_change_model_url,
            inputs=[form_model_url, global_state, form_seed_number, form_model_dropdown],
            outputs=[global_state, form_image_draw, form_image_mask_draw],
        )

        def on_click_run_projection(image_file, project_iterations_number, global_state):
            print("Done")
            print("Done")
            print("Done")
            print("Done")
            print("Done")
            drag_gan: DragGAN = get_model(global_state)
            trainable_latent = drag_gan.project(
                Image.open(image_file.name), num_steps=int(project_iterations_number), verbose=True
            )
            global_state["temporal_params"]["trainable_latent"] = trainable_latent

            image_raw = drag_gan.generate(trainable_latent)

            create_images(image_raw, global_state)

            return global_state, global_state["draws"]["image_with_points"], global_state["draws"]["image_with_mask"]

        form_update_image_project_btn.click(
            on_click_run_projection,
            inputs=[form_project_file, form_project_iterations_number, global_state],
            outputs=[global_state, form_image_draw, form_image_mask_draw],
        )

        def on_change_seed(seed, global_state):
            drag_gan: DragGAN = get_model(global_state)
            trainable_latent, image_raw = generate_from_seed(drag_gan, int(seed))

            create_images(image_raw, global_state)

            # Restart draw
            global_state["temporal_params"] = {"trainable_latent": trainable_latent}

            return global_state, image_raw, global_state["draws"]["image_with_mask"]

        form_seed_number.change(
            on_change_seed,
            inputs=[form_seed_number, global_state],
            outputs=[global_state, form_image_draw, form_image_draw],
        )

        def on_click_reset_image(global_state):
            global_state["images"]["image_raw"] = global_state["images"]["image_orig"].copy()
            global_state["draws"]["image_with_points"] = global_state["images"]["image_orig"].copy()

            return global_state, global_state["images"]["image_raw"], global_state["draws"]["image_with_mask"]

        form_reset_image.click(
            on_click_reset_image,
            inputs=[global_state],
            outputs=[global_state, form_image_draw, form_image_mask_draw],
        )

        # Update parameters
        def on_change_update_image_seed(seed, global_state):
            drag_gan: DragGAN = get_model(global_state)
            trainable_latent, image_raw = generate_from_seed(drag_gan, int(seed))

            create_images(image_raw, global_state)

            # Restart draw
            global_state["temporal_params"] = {"trainable_latent": trainable_latent}

            return global_state, image_raw, global_state["draws"]["image_with_mask"]

        form_update_image_seed_btn.click(
            on_change_update_image_seed,
            inputs=[form_seed_number, global_state],
            outputs=[global_state, form_image_draw, form_image_mask_draw],
        )

        # Tools tab listeners
        def on_change_dropdown_points(curr_point, global_state):
            global_state["curr_point"] = curr_point
            image_draw = draw_points_on_image(
                global_state["images"]["image_raw"],
                global_state["points"],
                global_state["curr_point"],
            )
            return global_state, image_draw

        form_points_dropdown.change(
            on_change_dropdown_points,
            inputs=[form_points_dropdown, global_state],
            outputs=[global_state, form_image_draw],
        )

        form_type_point_radio.change(
            partial(on_change_single_global_state, "curr_type_point"),
            inputs=[form_type_point_radio, global_state],
            outputs=[global_state],
        )

        # ==== Params
        form_lambda_number.change(
            partial(on_change_single_global_state, ["params", "motion_lambda"]),
            inputs=[form_lambda_number, global_state],
            outputs=[global_state],
        )

        def on_change_motion_lr(motion_lr, global_state):
            global_state["params"]["motion_lr"] = motion_lr

            return global_state

        form_motion_lr_number.change(
            on_change_motion_lr,
            inputs=[form_motion_lr_number, global_state],
            outputs=[global_state],
        )

        form_magnitude_direction_in_pixels_number.change(
            partial(
                on_change_single_global_state,
                ["params", "magnitude_direction_in_pixels"],
            ),
            inputs=[form_motion_lr_number, global_state],
            outputs=[global_state],
        )

        form_r1_in_pixels_number.change(
            partial(
                on_change_single_global_state,
                ["params", "r1_in_pixels"],
                map_transform=lambda x: int(x),
            ),
            inputs=[form_r1_in_pixels_number, global_state],
            outputs=[global_state],
        )

        form_r2_in_pixels_number.change(
            partial(
                on_change_single_global_state,
                ["params", "r2_in_pixels"],
                map_transform=lambda x: int(x),
            ),
            inputs=[form_r2_in_pixels_number, global_state],
            outputs=[global_state],
        )

        def on_click_start(global_state):
            p_in_pixels = []
            t_in_pixels = []
            valid_points = []

            # Prepare the points for the inference
            if len(global_state["points"]) == 0:
                image_draw = draw_points_on_image(
                    global_state["draws"]["image_with_points"],
                    global_state["points"],
                    global_state["curr_point"],
                )
                return global_state, 0, image_draw, gr.File.update(visible=False)

            # Transform the points into torch tensors
            for key_point, point in global_state["points"].items():
                try:
                    p_start = point.get("start_temp", point["start"])
                    p_end = point["target"]

                    if p_start is None or p_end is None:
                        continue

                except KeyError:
                    continue

                p_in_pixels.append(p_start)
                t_in_pixels.append(p_end)
                valid_points.append(key_point)

            p_in_pixels = torch.tensor(p_in_pixels)
            t_in_pixels = torch.tensor(t_in_pixels)

            r1_in_pixels = torch.tensor([global_state["params"]["r1_in_pixels"]]).float()
            r2_in_pixels = torch.tensor([global_state["params"]["r2_in_pixels"]]).float()

            # Mask for the paper:
            # M=1 that you want to edit
            # M=0 that you want to preserve
            mask_in_pixels = torch.tensor(global_state["images"]["image_mask"]).float()

            # Init the DragGAN
            drag_gan: DragGAN = get_model(global_state)
            trainable_latent = global_state["temporal_params"]["trainable_latent"]
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
            ) = drag_gan.init(
                trainable_latent=trainable_latent,
                p_in_pixels=p_in_pixels,
                r1_in_pixels=r1_in_pixels,
                r2_in_pixels=r2_in_pixels,
                t_in_pixels=t_in_pixels,
                magnitude_direction_in_pixels=global_state["params"]["magnitude_direction_in_pixels"],
                mask_in_pixels=mask_in_pixels,
                motion_lr=global_state["params"]["motion_lr"],
                optimizer=global_state["temporal_params"].get("optimizer", None),
            )
            global_state["temporal_params"]["stop"] = False

            # Start to iterate
            step_idx = 0
            while True:
                # Stop the iteration if the user press the button
                if global_state["temporal_params"]["stop"]:
                    break

                p = drag_gan.step(
                    optimizer=optimizer,
                    motion_lambda=global_state["params"]["motion_lambda"],
                    trainable_latent=trainable_latent,
                    F0=F0,
                    p_init=p_init,
                    p=p,
                    t=t,
                    r1=r1,
                    r1_interpolation_samples=global_state["params"]["r1_in_pixels"],
                    r2=r2,
                    r2_interpolation_samples=global_state["params"]["r2_in_pixels"],
                    M=M,
                    magnitude_direction=magnitude_direction,
                )

                if step_idx % global_state["draw_interval"] == 0:
                    # Unnormalize the p and t to create a visualization
                    p_in_pixels = drag_gan.norm_coord_to_pixel_coord(p)
                    t_in_pixels = drag_gan.norm_coord_to_pixel_coord(t)

                    # Move points in the global state
                    for key_point, p_i, t_i in zip(valid_points, p_in_pixels, t_in_pixels):
                        global_state["points"][key_point]["start_temp"] = p_i.tolist()
                        global_state["points"][key_point]["target"] = t_i.tolist()

                    # Generate the image
                    image_step_pil = drag_gan.generate(trainable_latent)
                    global_state["images"]["image_raw"] = image_step_pil

                    # Draw points on the image
                    # image_draw = draw_points_on_image(
                    #     image_step_pil,
                    #     global_state["points"],
                    #     global_state["curr_point"],
                    # )
                    create_images(image_step_pil, global_state)

                    yield (
                        global_state,
                        step_idx,
                        global_state["draws"]["image_with_points"],
                        global_state["draws"]["image_with_mask"],
                        gr.File.update(visible=False),
                    )

                # increate step
                step_idx += 1

            # Create the output result
            trainable_latent = global_state["temporal_params"]["trainable_latent"]
            image_result = drag_gan.generate(trainable_latent)

            create_images(image_result, global_state)

            fp = NamedTemporaryFile(suffix=".png", delete=False)
            image_result.save(fp, "PNG")

            yield (
                global_state,
                step_idx,
                global_state["draws"]["image_with_points"],
                global_state["draws"]["image_with_mask"],
                gr.File.update(visible=True, value=fp.name),
            )

        form_start_btn.click(
            on_click_start,
            inputs=[global_state],
            outputs=[global_state, form_steps_number, form_image_draw, form_image_mask_draw, form_download_result_file],
        )

        def on_click_stop(global_state):
            global_state["temporal_params"]["stop"] = True

            return global_state

        form_stop_btn.click(on_click_stop, inputs=[global_state], outputs=[global_state])

        form_draw_interval_number.change(
            partial(
                on_change_single_global_state,
                "draw_interval",
                map_transform=lambda x: int(x),
            ),
            inputs=[form_draw_interval_number, global_state],
            outputs=[global_state],
        )

        # Add & remove points
        def on_click_add_point(global_state):
            choices = list(global_state["points"].keys())
            if len(choices) > 0:
                max_choice = int(choices[-1])

            else:
                max_choice = -1

            max_choice = str(max_choice + 1)

            global_state["curr_point"] = max_choice
            global_state["points"][max_choice] = {"start": None, "target": None}
            choices = choices + [max_choice]
            return (
                gr.Dropdown.update(choices=choices, value=max_choice),
                global_state,
            )

        form_add_point_btn.click(
            on_click_add_point,
            inputs=[global_state],
            outputs=[form_points_dropdown, global_state],
        )

        def on_click_remove_point(global_state):
            choice = global_state["curr_point"]
            del global_state["points"][choice]

            choices = list(global_state["points"].keys())

            if len(choices) > 0:
                global_state["curr_point"] = choices[0]

            return (
                gr.Dropdown.update(choices=choices, value=choices[0]),
                global_state,
            )

        form_remove_point_btn.click(
            on_click_remove_point,
            inputs=[global_state],
            outputs=[form_points_dropdown, global_state],
        )

        # Mask
        def on_click_reset_mask(global_state):
            global_state["images"]["image_mask"] = np.ones(
                (
                    global_state["images"]["image_raw"].size[1],
                    global_state["images"]["image_raw"].size[0],
                ),
                dtype=np.uint8,
            )
            global_state["draws"]["image_with_mask"] = draw_mask_on_image(
                global_state["images"]["image_raw"], global_state["images"]["image_mask"]
            )
            return global_state, global_state["draws"]["image_with_mask"]

        form_reset_mask_btn.click(
            on_click_reset_mask,
            inputs=[global_state],
            outputs=[global_state, form_image_mask_draw],
        )

        form_radius_mask_number.change(
            partial(
                on_change_single_global_state,
                "radius_mask",
                map_transform=lambda x: int(x),
            ),
            inputs=[form_radius_mask_number, global_state],
            outputs=[global_state],
        )

        # Image
        def on_click_points_tab(global_state):
            global_state["curr_tool"] = "point"
            return (
                global_state,
                gr.Image.update(visible=True),
                gr.Image.update(visible=False),
            )

        points_tab.select(
            on_click_points_tab,
            inputs=[global_state],
            outputs=[global_state, form_image_draw, form_image_mask_draw],
        )

        def on_click_mask_tab(global_state):
            global_state["curr_tool"] = "mask"
            return (
                global_state,
                gr.Image.update(visible=False),
                gr.Image.update(visible=True),
            )

        mask_tab.select(
            on_click_mask_tab,
            inputs=[global_state],
            outputs=[global_state, form_image_draw, form_image_mask_draw],
        )

        def on_click_image(global_state, evt: gr.SelectData):
            xy = evt.index
            curr_point = global_state["curr_point"]
            if curr_point is None:
                return global_state, global_state["images"]["image_raw"]

            curr_type_point = global_state["curr_type_point"]
            if curr_type_point == "start (p)":
                curr_type_point = "start"

            elif curr_type_point == "target (t)":
                curr_type_point = "target"
            global_state["points"][curr_point][curr_type_point] = xy

            # Draw on image
            image_draw = draw_points_on_image(
                global_state["images"]["image_raw"],
                global_state["points"],
                global_state["curr_point"],
            )
            global_state["draws"]["image_with_points"] = image_draw

            return global_state, image_draw

        form_image_draw.select(
            on_click_image,
            inputs=[global_state],
            outputs=[global_state, form_image_draw],
        )

        def on_click_mask(global_state, evt: gr.SelectData):
            xy = evt.index

            radius_mask = int(global_state["radius_mask"])

            image_mask = np.uint8(255 * global_state["images"]["image_mask"])
            image_mask = cv2.circle(image_mask, xy, radius_mask, 0, -1) > 127
            global_state["images"]["image_mask"] = image_mask

            image_with_mask = draw_mask_on_image(global_state["images"]["image_raw"], image_mask)
            global_state["draws"]["image_with_mask"] = image_with_mask

            return global_state, image_with_mask

        form_image_mask_draw.select(
            on_click_mask,
            inputs=[global_state],
            outputs=[global_state, form_image_mask_draw],
        )

    return app


if __name__ == "__main__":
    import argparse
    import os

    default_network_pkl = os.environ.get("NETWORK_PKL")
    if default_network_pkl is None or default_network_pkl == "":
        default_network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl"

    default_seed = os.environ.get("SEED")
    if default_seed is None or default_seed == "":
        default_seed = 42

    default_device = os.environ.get("DEVICE")
    if default_device is None or default_device == "":
        default_device = "cuda:0"
    default_device = default_device if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Execute DragGAN using gradio GUI.")
    parser.add_argument(
        "--network_pkl", type=str, default=default_network_pkl, help="Path or url of the network pkl", required=False
    )
    parser.add_argument("--seed", type=int, default=default_seed, help="Default seed", required=False)
    parser.add_argument("--device", type=str, default=default_device, help="Device (cpu or cuda:index)", required=False)
    parser.add_argument("--share", action="store_true", help="Share gradio GUI", required=False)

    args = parser.parse_args()

    generator = StyleGANv2Generator(
        network_pkl=args.network_pkl,
    )

    app = main(
        generator=generator,
        default_seed=args.seed,
        device=args.device,
    )

    share = args.share | bool(os.environ.get("SHARE", False))

    gr.close_all()
    app.queue(concurrency_count=2, max_size=20).launch(share=share, server_name="0.0.0.0")
