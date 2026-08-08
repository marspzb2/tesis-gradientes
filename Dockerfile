ARG PYTHON_VERSION=3.14

FROM python:${PYTHON_VERSION}-slim AS cpp-builder

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        clang \
        clang-tools \
        cmake \
        git \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY CMakeLists.txt CMakePresets.json pyproject.toml README.md ./

RUN python -m pip install --no-cache-dir . pybind11

COPY src/cpp ./src/cpp

RUN CMAKE_PREFIX_PATH="$(python -c \
        'import os, pybind11, torch; print(os.pathsep.join((torch.utils.cmake_prefix_path, pybind11.get_cmake_dir())))')" \
    cmake --preset docker

RUN cmake --build --preset docker


FROM python:${PYTHON_VERSION}-slim AS runtime

ARG PYTHON_VERSION

WORKDIR /app

COPY pyproject.toml README.md ./
RUN python -m pip install --no-cache-dir .

COPY src/py ./src/py
COPY runServer.sh ./
COPY --from=cpp-builder /app/build/cpp/Docker/lib/compression_python*.so \
    /usr/local/lib/python${PYTHON_VERSION}/site-packages/

RUN chmod +x runServer.sh

ENV PYTHONUNBUFFERED=1

CMD ["python", "src/py/pytorch-data-parallel-training.py"]
