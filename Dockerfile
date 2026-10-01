# syntax=docker/dockerfile:1.7
# NVIDIA's unified PyTorch container, including its CUDA and PyTorch stack.
FROM nvcr.io/nvidia/pytorch:26.09-py3 AS ollama-builder

ARG OLLAMA_VERSION=0.35.0
ARG OLLAMA_BUILD_JOBS=1
ARG OLLAMA_CUDA_THREADS=1

# Build only the CUDA 13 backend needed by JetPack 7 / Orin (SM 8.7). This
# links against the CUDA toolkit provided by the NGC image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ccache clang cmake git ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "v${OLLAMA_VERSION}" \
    https://github.com/ollama/ollama.git /opt/ollama

WORKDIR /opt/ollama
RUN curl -fsSL "https://go.dev/dl/go$(awk '/^go / { print $2 }' go.mod).linux-arm64.tar.gz" \
    | tar -xz -C /usr/local

ENV PATH=/usr/local/go/bin:${PATH}
ENV CC=clang
ENV CXX=clang++
RUN --mount=type=cache,target=/root/.cache/go-build \
    --mount=type=cache,target=/root/.cache/ccache \
    cmake -S . -B build -G Ninja \
        -DOLLAMA_LLAMA_BACKENDS=cuda_v13 \
        -DOLLAMA_BUILD_PARALLEL="${OLLAMA_BUILD_JOBS}" \
        -DCMAKE_CUDA_FLAGS="-t ${OLLAMA_CUDA_THREADS}" \
        -DGGML_CPU_ALL_VARIANTS=OFF \
        -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
        -DCMAKE_CUDA_ARCHITECTURES=87 \
    && cmake --build build --parallel "${OLLAMA_BUILD_JOBS}"

FROM nvcr.io/nvidia/pytorch:26.09-py3

ARG PIP_DISABLE_PIP_VERSION_CHECK=1
ARG OLLAMA_VERSION=0.35.0

# Keep additions independent from packages and constraints provided by NGC.
COPY requirements-python.lock /opt/locks/
RUN --mount=type=cache,id=ngc-pip,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --ignore-installed --no-deps -r /opt/locks/requirements-python.lock

# Copy the source-built CLI and CUDA 13 backend. Model data remains in the
# runtime OLLAMA_MODELS volume rather than the image.
COPY --from=ollama-builder /opt/ollama/ollama /usr/bin/ollama
COPY --from=ollama-builder /opt/ollama/build/lib/ollama/ /usr/lib/ollama/
RUN rm -f /usr/lib/ollama/cuda_v13/libcublas.so.13 \
          /usr/lib/ollama/cuda_v13/libcublas.so.13.* \
          /usr/lib/ollama/cuda_v13/libcublasLt.so.13 \
          /usr/lib/ollama/cuda_v13/libcublasLt.so.13.* \
          /usr/lib/ollama/cuda_v13/libcudart.so.13 \
          /usr/lib/ollama/cuda_v13/libcudart.so.13.* \
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
