import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import gradio as gr
from drag_gan import DragGAN
from pathlib import Path
from threading import Thread
from functools import partial

def init_drag_gan(network_pkl, device, seed):
    drag_gan = DragGAN(network_pkl, device=device)
    w_latent = drag_gan.get_w_latent_from_seed(seed=seed)
    image_orig = drag_gan.generate(w_latent)
    return drag_gan, w_latent, image_orig

def init_wrapper_drag_gan(ref, global_state, seed):
    device = global_state['device']
    drag_gan, w_latent, image_raw = init_drag_gan(ref, device, int(seed))
    
    global_state['image_orig'] = image_raw.copy()
    global_state['image_raw'] = image_raw
    global_state['image_draw'] = image_raw.copy()

    global_state['image_mask'] = np.ones((image_raw.size[1], image_raw.size[0]), dtype=np.uint8)
    global_state['image_mask_draw'] = draw_mask_on_image(global_state['image_raw'], global_state['image_mask'])

    del global_state['restart_params']
    global_state['restart_params'] = {}

    global_state['model'] = drag_gan
    global_state['restart_params']['w_latent'] = w_latent
    
    # Restart draw
    return global_state, image_raw


font = ImageFont.truetype(str(Path(__file__).parent / "misc/Roboto-Medium.ttf"), 32)
def draw_points_on_image(image, points, curr_point = None):
    overlay_rgba = Image.new('RGBA', image.size, 0)
    overlay_draw = ImageDraw.Draw(overlay_rgba)
    for point_key, point in points.items():
        if curr_point is not None and curr_point == point_key:
            p_color = (255, 0, 0)
            t_color = (0, 0, 255)

        else:
            p_color = (255, 0, 0, 35)
            t_color = (0, 0, 255, 35)

        rad_draw = int(image.size[0] * 0.02)
        
        p_start = point.get('start_temp', point['start'])
        p_target = point['target']

        if p_start is not None and p_target is not None:
            p_draw = int(p_start[0]), int(p_start[1])
            t_draw = int(p_target[0]), int(p_target[1])

            overlay_draw.line((p_draw[0], p_draw[1], t_draw[0], t_draw[1]), fill=(255, 255, 0), width=2)

        if p_start is not None:
            p_draw = int(p_start[0]), int(p_start[1])
            overlay_draw.ellipse(
                (p_draw[0] - rad_draw, p_draw[1] - rad_draw, p_draw[0] + rad_draw, p_draw[1] + rad_draw), 
                fill=p_color
            )

            if curr_point is not None and curr_point == point_key:
                overlay_draw.text(p_draw, "p", font=font, align ="center", fill=(0, 0, 0))
        
        if p_target is not None:
            t_draw = int(p_target[0]), int(p_target[1])
            overlay_draw.ellipse(
                (t_draw[0] - rad_draw, t_draw[1] - rad_draw, t_draw[0] + rad_draw, t_draw[1] + rad_draw), 
                fill=t_color
            )
            
            if curr_point is not None and curr_point == point_key:
                overlay_draw.text(t_draw, "t", font=font, align ="center", fill=(0, 0, 0))

    return Image.alpha_composite(image.convert('RGBA'), overlay_rgba).convert('RGB')

def draw_mask_on_image(image, mask):
    im_mask = np.uint8(mask * 255)
    im_mask_rgba = np.concatenate((
        np.tile(im_mask[..., None], [1, 1, 3]),
        45 * np.ones((im_mask.shape[0], im_mask.shape[1], 1), dtype=np.uint8)
    ), axis=-1)
    im_mask_rgba = Image.fromarray(im_mask_rgba).convert('RGBA')

    return Image.alpha_composite(image.convert('RGBA'), im_mask_rgba).convert('RGB')


def main(
    network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl",
    default_seed: int = 42,
    device: str = 'cuda:0'
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
        drag_gan, w_latent, image_orig = init_drag_gan(network_pkl, device, default_seed)

        global_state = gr.State({
            'restart_params': {
                'w_latent': w_latent,
            },

            'params': {
                'motion_lr': 2e-3,
                'motion_lambda': 20,
                'trainable_w_dims': 6,
                
                'r1_in_pixels': 3,
                'r2_in_pixels': 12
            },


            'device': device,
            'image_orig': image_orig.copy(),
            'image_raw': image_orig,

            'image_mask': np.ones((image_orig.size[1], image_orig.size[0]), dtype=np.uint8),

            'draw_interval': 5,
            'radius_mask': 51,
            'projection_steps': 1_000,

            'model': drag_gan,

            'points': {
            },
            'curr_point': None,
            'curr_type_point': 'start',

            # opt params
            

            'restart': True,
        })
        image_draw = draw_points_on_image(global_state.value['image_raw'], global_state.value['points'], global_state.value['curr_point'])
        global_state.value['image_draw'] = image_draw
        
        image_mask_draw = draw_mask_on_image(global_state.value['image_raw'], global_state.value['image_mask'])
        global_state.value['image_mask_draw'] = image_draw


        with gr.Row():
            # Left column
            with gr.Column(scale=.7):
                with gr.Accordion("Information"):
                    with gr.Tab("Tutorial"):
                        gr.Markdown(tutorial)
                    with gr.Tab("About the paper"):
                        gr.Markdown(about_paper)

                with gr.Accordion("Network & latent"):
                    form_model_pickle_file = gr.File(label="Pickle file")

                    with gr.Row():
                        form_model_url = gr.Textbox(placeholder="Url of the pkl", label="Url")
                        form_model_url_btn = gr.Button("Submit")

                    with gr.Row():
                        form_project_file = gr.File(label="Image project file")
                        form_project_iterations = gr.Number(value=global_state.value['projection_steps'], label="Image projection num steps")

                    with gr.Row():    
                        form_seed_number = gr.Number(value=default_seed, interactive=True, label="Seed")
                        form_reset_image = gr.Button("Reset image")

                with gr.Accordion("Tools"):
                    with gr.Tab("Pair-Points") as points_tab:
                        form_dropdown_points = gr.Dropdown(choices=[], value="", interactive=True, label="List of pair-points")
                        
                        form_type_point = gr.Radio(["start (p)", "target (t)"], value="start (p)", label="Type")

                        with gr.Row():
                            form_add_point_btn = gr.Button("Add pair-point").style(full_width=True)
                            form_remove_point_btn = gr.Button("Remove pair-point").style(full_width=True)

                    with gr.Tab("Mask (subtractive mask)") as mask_tab:
                        gr.Markdown("""
                            White zone = editable by DragGAN
                            Transparent zone = not editable by DragGAN.
                        """)
                        form_reset_mask_btn = gr.Button("Reset mask").style(full_width=True)
                        form_radius_mask_number = gr.Number(value=global_state.value['radius_mask'], interactive=True, label="Radius (pixels)").style(full_width=False)              

                    with gr.Row():
                        form_start_btn = gr.Button("Start").style(full_width=True)
                        form_stop_btn = gr.Button("Stop").style(full_width=True)
                        form_steps = gr.Number(value=0, label="Steps", interactive=False).style(full_width=False)
                        form_draw_interval = gr.Number(value=global_state.value['draw_interval'], label="Draw Interval (steps)", interactive=False).style(full_width=False)

                    with gr.Row():
                        form_lambda_number = gr.Number(value=global_state.value['params']['motion_lambda'], interactive=True, label="Lambda").style(full_width=True)
                        form_trainable_w_dims = gr.Number(value=global_state.value['params']['trainable_w_dims'], interactive=True, label="Trainable W latent dims").style(full_width=True)
                        form_motion_lr = gr.Number(value=global_state.value['params']['motion_lr'], interactive=True, label="LR").style(full_width=True)
                            
                    with gr.Row():
                        form_r1_in_pixels_number = gr.Number(value=global_state.value['params']['r1_in_pixels'], interactive=True, label="R1 (pixels)").style(full_width=False)
                        form_r2_in_pixels_number = gr.Number(value=global_state.value['params']['r2_in_pixels'], interactive=True, label="R2 (pixels)").style(full_width=False)
                        
            # Right column 
            with gr.Column():
                form_image_draw = gr.Image(image_draw, elem_classes="image_nonselectable")
                form_mask_draw = gr.Image(image_mask_draw, visible=False, elem_classes="image_nonselectable")
                gr.Markdown("Credits: Adrià Ciurana Lanau | info@dreamlearning.ai")

                

        # Network & latents tab listeners
        def on_change_model_pickle(model_pickle_file, global_state, seed):
            return init_wrapper_drag_gan(model_pickle_file.name, global_state, seed)
        form_model_pickle_file.change(on_change_model_pickle, inputs=[form_model_pickle_file, global_state, form_seed_number], outputs=[global_state, form_image_draw])

        def on_change_model_url(url, global_state, seed):
            return init_wrapper_drag_gan(url, global_state, seed)
        form_model_url_btn.click(on_change_model_url, inputs=[form_model_url, global_state, form_seed_number], outputs=[global_state, form_image_draw])   

        def on_change_project_file(image_file, global_state):
            drag_gan: DragGAN = global_state["model"]
            num_steps = global_state["projection_steps"]
            w_latent = drag_gan.project(Image.open(image_file.name), num_steps=num_steps, verbose=True)
            
            image_raw = drag_gan.generate(w_latent)
            global_state['image_orig'] = image_raw.copy()
            global_state['image_raw'] = image_raw
            global_state['image_draw'] = image_raw.copy()

            global_state['image_mask'] = np.ones((image_raw.size[1], image_raw.size[0]), dtype=np.uint8)
            global_state['image_mask_draw'] = draw_mask_on_image(global_state['image_raw'], global_state['image_mask'])

            return global_state, global_state['image_draw']

        form_project_file.change(on_change_project_file, inputs=[form_project_file, global_state], outputs=[global_state, form_image_draw])

        def on_project_iterations(form_project_iterations, global_state):
            global_state["projection_steps"] = form_project_iterations
            return global_state
        form_project_iterations.change(on_project_iterations, inputs=[form_project_iterations, global_state], outputs=[global_state]) 


        def on_change_seed(seed, global_state):
            drag_gan: DragGAN = global_state['model']
            w_latent = drag_gan.get_w_latent_from_seed(int(seed))
            image_raw = drag_gan.generate(w_latent)
   
            global_state['image_orig'] = image_raw.copy()
            global_state['image_raw'] = image_raw
            global_state['image_draw'] = image_raw.copy()

            global_state['image_mask'] = np.ones((image_raw.size[1], image_raw.size[0]), dtype=np.uint8)
            global_state['image_mask_draw'] = draw_mask_on_image(global_state['image_raw'], global_state['image_mask'])

            del global_state['restart_params']
            global_state['restart_params'] = {}
            global_state['restart_params']['w_latent'] = w_latent
            # Restart draw

            return global_state, image_raw
        form_seed_number.change(on_change_seed, inputs=[form_seed_number, global_state], outputs=[global_state, form_image_draw]) 

        def on_click_reset_image(global_state):
            global_state['image_raw'] = global_state['image_orig'].copy()
            global_state['image_draw'] = global_state['image_orig'].copy()

            return global_state, global_state['image_raw']
        form_reset_image.click(on_click_reset_image, inputs=[global_state], outputs=[global_state, form_image_draw])       

        # Tools tab listeners
        def on_change_dropdown_points(curr_point, global_state):
            global_state["curr_point"] = curr_point
            image_draw = draw_points_on_image(global_state['image_raw'], global_state['points'], global_state['curr_point'])
            return global_state, image_draw
        form_dropdown_points.change(on_change_dropdown_points, inputs=[form_dropdown_points, global_state], outputs=[global_state, form_image_draw])

        def on_change_type_point(type_point, global_state):
            global_state["curr_type_point"] = type_point
            return global_state

        form_type_point.change(on_change_type_point, inputs=[form_type_point, global_state], outputs=[global_state])


        # ====xxx
        def on_change_lambda(motion_lambda, global_state):
            global_state['params']['motion_lambda'] = motion_lambda

            return global_state
        form_lambda_number.change(on_change_lambda, inputs=[form_lambda_number, global_state], outputs=[global_state])

        def on_change_trainable_w_dims(trainable_w_dims, global_state):
            global_state['params']['trainable_w_dims'] = int(trainable_w_dims)

            return global_state
        form_trainable_w_dims.change(on_change_trainable_w_dims, inputs=[form_trainable_w_dims, global_state], outputs=[global_state])

        def on_change_motion_lr(motion_lr, global_state):
            global_state['params']['motion_lr'] = motion_lr

            return global_state
        form_motion_lr.change(on_change_motion_lr, inputs=[form_motion_lr, global_state], outputs=[global_state])

        def on_change_r1(r1_in_pixels, global_state):
            global_state['params']['r1_in_pixels'] = int(r1_in_pixels)

            return global_state
        form_r1_in_pixels_number.change(on_change_r1, inputs=[form_r1_in_pixels_number, global_state], outputs=[global_state])

        def on_change_r2(r2_in_pixels, global_state):
            global_state['params']['r2_in_pixels'] = int(r2_in_pixels)

            return global_state
        form_r2_in_pixels_number.change(on_change_r2, inputs=[form_r2_in_pixels_number, global_state], outputs=[global_state])

        def on_click_start(global_state):
            p_in_pixels = []
            t_in_pixels = []

            valid_points = []

            if len(global_state['points']) == 0:
                image_draw = draw_points_on_image(global_state['image_draw'], global_state['points'], global_state['curr_point'])
                return global_state, 0, image_draw


            for key_point, point in global_state['points'].items():
                try:
                    p_start = point.get('start_temp', point['start'])
                    p_end = point['target']

                    p_in_pixels.append(p_start)
                    t_in_pixels.append(p_end)
                    valid_points.append(key_point)

                except KeyError:
                    continue

            p_in_pixels = torch.tensor(p_in_pixels)
            t_in_pixels = torch.tensor(t_in_pixels)

            r1_in_pixels = torch.tensor([global_state['params']['r1_in_pixels']]).float()
            r2_in_pixels = torch.tensor([global_state['params']['r2_in_pixels']]).float()

            # Mask for the paper:
            # M=1 that you want to edit
            # M=0 that you want to preserve
            mask_in_pixels = torch.tensor(global_state['image_mask']).float()
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
                F0
            ) = drag_gan.init(
                w_latent=global_state['restart_params']['w_latent'],
                trainable_w_dims=global_state['params']['trainable_w_dims'],
                p_in_pixels=p_in_pixels,
                r1_in_pixels=r1_in_pixels,
                r2_in_pixels=r2_in_pixels,
                t_in_pixels=t_in_pixels,
                magnitude_direction_in_pixels=1, # TODO
                mask_in_pixels=mask_in_pixels, # TODO
                motion_lr=global_state['params']['motion_lr'],
                optimizer=global_state['restart_params'].get('optimizer', None),
            )

            global_state["restart_params"]["stop"] = False
            step_idx = 0
            while True:
                if global_state['restart_params']['stop']:
                    break

                p = drag_gan.step(
                    optimizer=optimizer,
                    motion_lambda=global_state['params']['motion_lambda'],

                    w_latent_learn=w_latent_learn,
                    w_latent_fix=w_latent_fix,

                    F0=F0,
                    p_init=p_init,
                    p=p,
                    t=t,
                    r1=r1,
                    r1_interpolation_samples=global_state['params']['r1_in_pixels'],
                    r2=r2,
                    r2_interpolation_samples=global_state['params']['r2_in_pixels'],
                    M=M,
                    magnitude_direction=magnitude_direction,
                )
                
                p_in_pixels = drag_gan.norm2pixel(p)
                t_in_pixels = drag_gan.norm2pixel(t)

                for key_point, p_i, t_i in zip(valid_points, p_in_pixels, t_in_pixels):
                    global_state["points"][key_point]["start_temp"] = p_i.tolist()
                    global_state["points"][key_point]["target"] = t_i.tolist()

                img_step_pil = drag_gan.generate_image_from_split_w_latent(w_latent_learn, w_latent_fix)
                img_step_pil = drag_gan.draw_p_image(img_step_pil, p, t)  

                global_state['image_raw'] = img_step_pil
                image_draw = draw_points_on_image(img_step_pil, global_state['points'], global_state['curr_point'])

                if step_idx % global_state["draw_interval"] == 0:
                    yield global_state, step_idx, image_draw
                step_idx += 1
        
        form_start_btn.click(on_click_start, inputs=[global_state], outputs=[global_state, form_steps, form_image_draw])

        def on_click_stop(global_state):
            global_state["restart_params"]["stop"] = True
            return global_state
        form_stop_btn.click(on_click_stop, inputs=[global_state], outputs=[global_state])

        def on_change_interval(global_state, draw_interval):
            global_state["draw_interval"] = draw_interval
            return global_state
        form_draw_interval.change(on_change_interval, inputs=[global_state], outputs=[global_state])

        # Add & remove points
        def on_click_add_point(global_state):
            choices = list(global_state["points"].keys())
            if len(choices) > 0:
                max_choice = int(choices[-1])
                
            else:
                max_choice = -1

            max_choice = str(max_choice + 1)

            global_state["curr_point"] = max_choice
            global_state["points"][max_choice] = {'start': None, 'target': None}
            choices = choices + [max_choice]
            return gr.Dropdown.update(choices=choices, value=max_choice), global_state
        form_add_point_btn.click(on_click_add_point, inputs=[global_state], outputs=[form_dropdown_points, global_state])

        def on_click_remove_point(global_state):
            choice = global_state['curr_point']
            del global_state['points'][choice]
            
            choices = list(global_state['points'].keys())
            global_state["curr_point"] = choices[0]
            return gr.Dropdown.update(choices=choices, value=choices[0]), global_state
        form_remove_point_btn.click(on_click_remove_point, inputs=[global_state], outputs=[form_dropdown_points, global_state])

        # Mask
        def on_click_reset_mask(global_state): 
            global_state['image_mask'] = np.ones((global_state['image_raw'].size[1], global_state['image_raw'].size[0]), dtype=np.uint8)
            global_state['image_mask_draw'] = draw_mask_on_image(global_state['image_raw'], global_state['image_mask'])
            return global_state, global_state['image_mask_draw']
        form_reset_mask_btn.click(on_click_reset_mask, inputs=[global_state], outputs=[global_state, form_mask_draw])


        def on_change_radius_mask(radius_mask, global_state):
            global_state['radius_mask'] = radius_mask
            return global_state
        form_radius_mask_number.change(on_change_radius_mask, inputs=[form_radius_mask_number, global_state], outputs=[global_state])

        # Image
        def on_click_points_tab(global_state):
            global_state["curr_tool"] = "point"
            return global_state, gr.Image.update(visible=True), gr.Image.update(visible=False)
        points_tab.select(on_click_points_tab, inputs=[global_state], outputs=[global_state, form_image_draw, form_mask_draw])  
        
        def on_click_mask_tab(global_state):
            global_state["curr_tool"] = "mask"
            return global_state, gr.Image.update(visible=False), gr.Image.update(visible=True)
        mask_tab.select(on_click_mask_tab, inputs=[global_state], outputs=[global_state, form_image_draw, form_mask_draw])    
       
        def on_click_image(global_state, evt: gr.SelectData):
            xy = evt.index
            curr_point = global_state['curr_point']
            if curr_point is None:
                return global_state, global_state['image_raw']
            
            curr_type_point = global_state['curr_type_point']
            if curr_type_point == 'start (p)':
                curr_type_point = 'start'
            
            elif curr_type_point == 'target (t)':
                curr_type_point = 'target'
            global_state['points'][curr_point][curr_type_point] = xy
            
            # Draw on image
            image_draw = draw_points_on_image(global_state['image_raw'], global_state['points'], global_state['curr_point'])
            global_state['image_draw'] = image_draw

            return global_state, image_draw
        form_image_draw.select(on_click_image, inputs=[global_state], outputs=[global_state, form_image_draw])

        def on_click_mask(global_state, evt: gr.SelectData):
            xy = evt.index

            radius_mask = int(global_state['radius_mask'])

            image_mask = np.uint8(255 * global_state['image_mask'])
            image_mask = cv2.circle(image_mask, xy, radius_mask, 0, -1) > 127
            global_state['image_mask'] = image_mask

            image_mask_draw = draw_mask_on_image(global_state['image_raw'], image_mask)
            global_state['image_mask_draw'] = image_mask_draw

            return global_state, image_mask_draw
        form_mask_draw.select(on_click_mask, inputs=[global_state], outputs=[global_state, form_mask_draw])
        
        # Refresh data
        # app.run_forever(on_refresh, inputs=[global_state], outputs=[form_image_draw, form_steps], every=gr.TimeDelta.Seconds * 0.2)
    return app

if __name__ == '__main__':
    import fire
    app = main()
    gr.close_all()
    fire.Fire(app.queue(concurrency_count=2, max_size=20).launch)