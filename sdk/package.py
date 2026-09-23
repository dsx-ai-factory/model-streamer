#!/usr/bin/env python3
"""Package the current architecture's Bazel outputs without rebuilding them."""

import argparse
import hashlib
import io
import re
import struct
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = {
    "libstreamer.so": "streamer",
    "libstreamers3.so": "s3",
    "libstreamergcs.so": "gcs",
    "libstreamerazure.so": "azure",
}
MACHINES = {"x86_64": 62, "aarch64": 183}


def validate_library(path, arch):
    # Inspect the ELF header as well as the filename: bazel-bin changes when the
    # build switches architectures, and silently shipping the other ISA is easy.
    with path.open("rb") as source:
        header = source.read(20)
    if (len(header) != 20 or header[:6] != b"\x7fELF\x02\x01"
            or struct.unpack_from("<HH", header, 16) != (3, MACHINES[arch])):
        raise ValueError("{} is not an ELF64 {} shared library".format(path, arch))
    dynamic = subprocess.check_output(["readelf", "-dW", str(path)], text=True)
    if path.name == "libstreamer.so" and not re.search(
        r"\((?:RUNPATH|RPATH)\).*\[\$ORIGIN\]", dynamic
    ):
        raise ValueError("libstreamer.so must find its plugins through $ORIGIN")


def package(version, arch, bazel_bin, output):
    if version.startswith("v"):
        version = version[1:]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)", version):
        raise ValueError("Expected a release version such as 0.17.0 or v0.17.0")
    name = "runai-model-streamer-sdk-{}-linux-{}".format(version, arch)
    files = {}
    for library, target in LIBRARIES.items():
        source = bazel_bin / target / library
        validate_library(source, arch)
        files["lib/" + library] = source
    # Match the public include spelling established in #193: <streamer/streamer.h>.
    for header in ("streamer.h", "submission_id.h", "response_code.h", "device.h"):
        files["include/streamer/" + header] = ROOT / "cpp/streamer/api/streamer" / header
    files["LICENSE"] = ROOT / "LICENSE"
    files["README.md"] = ROOT / "docs/cpp_sdk.md"
    for filename in ("hello_streamer.c", "CMakeLists.txt"):
        files["examples/hello_streamer/" + filename] = ROOT / "sdk/examples/hello_streamer" / filename
    generated = {"VERSION": version + "\n"}
    for destination in (
        "lib/pkgconfig/runai-streamer.pc",
        "lib/cmake/runai-streamer/runai-streamer-config.cmake",
        "lib/cmake/runai-streamer/runai-streamer-config-version.cmake",
    ):
        template = ROOT / "sdk/templates" / (Path(destination).name + ".in")
        generated[destination] = template.read_text().replace("@VERSION@", version)

    output.mkdir(parents=True, exist_ok=True)
    archive = output / (name + ".tar.gz")
    with tarfile.open(archive, "w:gz", dereference=True) as tar:
        for destination, source in sorted(files.items()):
            tar.add(source, arcname=name + "/" + destination)
        for destination, contents in sorted(generated.items()):
            data = contents.encode()
            info = tarfile.TarInfo(name + "/" + destination)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    archive.with_name(archive.name + ".sha256").write_text(
        "{}  {}\n".format(digest.hexdigest(), archive.name)
    )
    print(archive)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--arch", required=True, choices=MACHINES)
    parser.add_argument("--bazel-bin", type=Path, default=ROOT / "cpp/bazel-bin")
    parser.add_argument("--output", type=Path, default=ROOT / "sdk/dist")
    args = parser.parse_args()
    package(args.version, args.arch, args.bazel_bin, args.output)
