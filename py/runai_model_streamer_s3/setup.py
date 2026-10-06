import os
from setuptools import setup, find_packages

VERSION = os.getenv("PACKAGE_VERSION", "0.0.0")
LIB = "libstreamers3.so"


def assert_lib_exists():
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LIB)
    if os.path.islink(lib_path):
        target_path = os.path.realpath(lib_path)
        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"{target_path} (target of the symlink) not found. Aborting build."
            )
    else:
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"{lib_path} not found. Aborting build.")


assert_lib_exists()
setup(
    name="runai-model-streamer-s3",
    version=VERSION,
    python_requires=">=3.12",
    license_files=("LICENSE",),
    packages=find_packages(),
    install_requires=["boto3"],
    data_files=[("/runai_model_streamer/libstreamer/lib/", [LIB])],
)
