# syntax=docker/dockerfile:1.7
# NVIDIA's unified PyTorch container, including its CUDA and PyTorch stack.
FROM nvcr.io/nvidia/pytorch:26.09-py3

ARG PIP_DISABLE_PIP_VERSION_CHECK=1
ARG OLLAMA_VERSION=0.35.0

# Keep additions independent from packages and constraints provided by NGC.
COPY requirements-python.lock /opt/locks/
RUN --mount=type=cache,id=ngc-pip,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --ignore-installed --no-deps -r /opt/locks/requirements-python.lock

# Install the ARM64 Ollama runtime without downloading models. Models are kept
# in a runtime volume (OLLAMA_MODELS) so they do not become part of the image.
RUN curl -fsSL "https://ollama.com/download/ollama-linux-arm64.tar.zst?version=${OLLAMA_VERSION}" \
    -o /tmp/ollama.tar.zst \
    && apt-get update \
    && apt-get install -y --no-install-recommends zstd \
    && tar --zstd -C /usr -xf /tmp/ollama.tar.zst \
    && rm -rf /usr/lib/ollama/cuda_v12 \
    && rm -f /usr/lib/ollama/cuda_v13/libcublas.so.13 \
             /usr/lib/ollama/cuda_v13/libcublas.so.13.* \
             /usr/lib/ollama/cuda_v13/libcublasLt.so.13 \
             /usr/lib/ollama/cuda_v13/libcublasLt.so.13.* \
             /usr/lib/ollama/cuda_v13/libcudart.so.13 \
             /usr/lib/ollama/cuda_v13/libcudart.so.13.* \
    && apt-get purge -y zstd \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm /tmp/ollama.tar.zst \
    && test -x /usr/bin/ollama \
    && test -f /usr/lib/ollama/cuda_v13/libggml-cuda.so

COPY scripts/ /opt/image-checks/
RUN python3 /opt/image-checks/verify-build.py

# Allow opt-in Jupyter startup while retaining NGC's original entrypoint.
COPY --chmod=755 scripts/start-container.sh /usr/local/bin/start-container
ENTRYPOINT ["/usr/local/bin/start-container"]
CMD ["bash"]

WORKDIR /workspace
EXPOSE 8888
EXPOSE 11434
