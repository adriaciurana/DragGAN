docker build -t drag_gan_gpu -f Dockerfile .. \
    --build-arg VERSION="2.0.1-cuda11.7-cudnn8-devel"