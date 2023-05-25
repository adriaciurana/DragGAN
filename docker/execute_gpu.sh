docker run \
    --name drag_gan_gpu_exec \
    --gpus all \
    --rm \
    -t \
    -p 7860:7860 \
    drag_gan_gpu
