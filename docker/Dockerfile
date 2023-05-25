ARG VERSION=latest

FROM pytorch/pytorch:${VERSION}
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHONUNBUFFERED=0

ENV NETWORK_PKL=
ENV SEED=
ENV DEVICE=
ENV SHARE=
EXPOSE 7860

WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg libsm6 libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/drag_gan
RUN pip install -r drag_gan/requirements.txt

WORKDIR /app/drag_gan

ENTRYPOINT ["python", "gradio_app.py"]
