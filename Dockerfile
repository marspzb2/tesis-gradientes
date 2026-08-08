ARG PYTHON_VERSION=3.14
ARG CUDA_VERSION=12.8.1
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

FROM ghcr.io/astral-sh/uv:latest AS uv

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS cpp-builder

ARG PYTHON_VERSION
ARG PYTORCH_INDEX_URL

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        clang \
        clang-tools \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

RUN uv venv --python "${PYTHON_VERSION}" /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

COPY CMakeLists.txt CMakePresets.json pyproject.toml README.md ./

RUN uv pip install \
        --python /opt/venv/bin/python \
        --index-url "${PYTORCH_INDEX_URL}" \
        torch==2.9.1 \
        torchvision==0.24.1

RUN uv pip install \
        --python /opt/venv/bin/python \
        cmake \
        ninja \
        pybind11 \
        .

COPY src/cpp ./src/cpp

RUN CMAKE_PREFIX_PATH="$(python -c \
        'import os, pybind11, torch; print(os.pathsep.join((torch.utils.cmake_prefix_path, pybind11.get_cmake_dir())))')" \
    cmake --preset docker

RUN cmake --build --preset docker


FROM python:${PYTHON_VERSION}-slim AS runtime

ARG PYTHON_VERSION
ARG PYTORCH_INDEX_URL

WORKDIR /app

COPY pyproject.toml README.md ./

RUN python -m pip install --no-cache-dir \
        --index-url "${PYTORCH_INDEX_URL}" \
        torch==2.9.1 \
        torchvision==0.24.1

RUN python -m pip install --no-cache-dir .

COPY src/py ./src/py
COPY runServer.sh ./
COPY --from=cpp-builder /app/build/cpp/Docker/lib/compression_python*.so \
    /app/native/

RUN chmod +x runServer.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/native

CMD ["python", "src/py/pytorch-data-parallel-training.py"]
