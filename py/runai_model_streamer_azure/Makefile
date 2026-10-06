ARCH := $(shell uname -m)
PLATFORM := manylinux_2_28_${ARCH}
PACKAGE_VERSION ?= 0.0.0

.PHONY: build test install_dev_dep clean

install_dev_dep:
	python3 -m pip install --require-hashes --only-binary=:all: -r ../../.devcontainer/dependencies/requirements.lock

build:
	python3 setup.py bdist_wheel --plat-name ${PLATFORM}

test: install_dev_dep
	python3 -m unittest discover -b runai_model_streamer_azure

clean:
	rm -rf build/ dist/ runai_model_streamer_azure.egg-info/

install:
	pip3 uninstall -y runai_model_streamer_azure && \
	pip3 install dist/runai_model_streamer_azure-${PACKAGE_VERSION}-py3-none-${PLATFORM}.whl
