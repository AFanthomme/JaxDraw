FROM ghcr.io/nvidia/jax:jax-2026-02-27

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN python -m pip install -r requirements.txt 
