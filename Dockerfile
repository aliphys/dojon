# syntax=docker/dockerfile:1.7
# NVIDIA's unified PyTorch container, including its CUDA and PyTorch stack.
FROM nvcr.io/nvidia/pytorch:26.09-py3

ARG PIP_DISABLE_PIP_VERSION_CHECK=1

# Keep additions independent from packages and constraints provided by NGC.
COPY requirements-python.lock /opt/locks/
RUN --mount=type=cache,id=ngc-pip,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --ignore-installed --no-deps -r /opt/locks/requirements-python.lock

COPY scripts/ /opt/image-checks/
RUN python3 /opt/image-checks/verify-build.py

# Allow opt-in Jupyter startup while retaining NGC's original entrypoint.
COPY --chmod=755 scripts/start-container.sh /usr/local/bin/start-container
ENTRYPOINT ["/usr/local/bin/start-container"]
CMD ["bash"]

WORKDIR /workspace
EXPOSE 8888
