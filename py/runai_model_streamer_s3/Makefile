ARCH := $(shell uname -m)
PLATFORM := manylinux_2_28_${ARCH}
PACKAGE_VERSION ?= 0.0.0

.PHONY: build clean

build:
	python3 setup.py bdist_wheel --plat-name ${PLATFORM}

test:
	python3 -m unittest discover -b runai_model_streamer_s3

clean:
	rm -rf build/ dist/ runai_model_streamer_s3.egg-info/

install:
	pip3 uninstall -y runai_model_streamer_s3 && \
	pip3 install dist/runai_model_streamer_s3-${PACKAGE_VERSION}-py3-none-${PLATFORM}.whl
