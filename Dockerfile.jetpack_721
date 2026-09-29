# syntax=docker/dockerfile:1.7
# JetPack libraries are the compatibility anchor; update this version deliberately.
FROM whitesscott/l4t-jetpack:r39.2.1

# Build-only pip settings; the base Python and NVIDIA libraries stay available.
ARG PIP_BREAK_SYSTEM_PACKAGES=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

# Keep the large, tested GPU dependency set independently cached.
COPY requirements-torch.lock /opt/locks/
RUN --mount=type=cache,id=jp721-pip,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --no-deps -r /opt/locks/requirements-torch.lock

COPY scripts/configure-libraries.py /opt/image-checks/
RUN python3 /opt/image-checks/configure-libraries.py

# Includes jtop once, from the pinned source matching the host service.
# Its build uses the locked setuptools and base wheel, without an isolated resolver.
COPY requirements-python.lock /opt/locks/
RUN --mount=type=cache,id=jp721-pip,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --no-deps --no-build-isolation \
        -r /opt/locks/requirements-python.lock

COPY scripts/ /opt/image-checks/
RUN python3 /opt/image-checks/verify-build.py

# Preserve the base CUDA setup while allowing opt-in Jupyter startup.
COPY --chmod=755 scripts/start-container.sh /usr/local/bin/start-container
ENTRYPOINT ["/usr/local/bin/start-container"]
CMD ["bash"]

WORKDIR /workspace
EXPOSE 8888
