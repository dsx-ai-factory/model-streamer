import os
from setuptools import setup, find_packages

VERSION = os.getenv("PACKAGE_VERSION", "0.0.0")
LIB = "libstreamer/lib/libstreamer.so"


def assert_lib_exists():
    lib_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "runai_model_streamer", LIB
    )
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
    name="runai-model-streamer",
    version=VERSION,
    python_requires=">=3.12",
    license_files=("LICENSE",),
    packages=find_packages(),
    package_data={"runai_model_streamer": [LIB]},
    install_requires=["torch>=2.0.0, <3.0.0", "humanize", "numpy"],
    extras_require={
        "s3": [f"runai_model_streamer_s3=={VERSION}"],
        "gcs": [f"runai_model_streamer_gcs=={VERSION}"],
        "azure": [f"runai_model_streamer_azure=={VERSION}"],
    },
)
