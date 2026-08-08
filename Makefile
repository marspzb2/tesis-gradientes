IMAGE_VERSION := 1.0.0
IMAGE_REPO := docker.io/marspzb/tesis-gradientes
#quay.io/marspzb/tesis-gradientes
.PHONY: clean hello activate
hello:
	echo "Hello"
init-cpp:
	cmake --preset default
build-cpp:
	cmake --build build/cpp/Release
install-py-env:
	uv venv --python 3.14
	echo "${PWD}/build/cpp/Release/lib" > .venv/lib/python3.14/site-packages/compression_python.pth
install-py:
	source "${PWD}/.venv/bin/activate"
	uv sync
activate:
	source "${PWD}/.venv/bin/activate"
build-image:
	podman build  --platform linux/amd64 -t $(IMAGE_REPO):latest -t  $(IMAGE_REPO):$(IMAGE_VERSION) --build-arg PYTHON_VERSION=3.14 -t tesis-gradientes .
push-image:
	podman push $(IMAGE_REPO):latest
	podman push $(IMAGE_REPO):$(IMAGE_VERSION)
clean:
	rm -rf build
	rm -rf out