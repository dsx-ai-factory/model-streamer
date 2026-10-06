"""Fail CI if a wheel violates our glibc 2.28 / architecture contract."""

import glob
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def check(wheel):
    arch = "aarch64" if wheel.name.endswith("_aarch64.whl") else "x86_64"
    expected_machine = "AArch64" if arch == "aarch64" else "Advanced Micro Devices X86-64"
    if not wheel.name.endswith(f"-manylinux_2_28_{arch}.whl"):
        raise ValueError(f"Unexpected wheel platform: {wheel.name}")
    libraries = 0
    with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory() as directory:
        for name in archive.namelist():
            data = archive.read(name)
            if not data.startswith(b"\x7fELF"):
                continue
            libraries += 1
            path = Path(directory) / f"library-{libraries}.so"
            path.write_bytes(data)
            header = subprocess.check_output(["readelf", "-h", str(path)], text=True)
            if expected_machine not in header:
                raise ValueError(f"Wrong architecture in {wheel.name}: {name}")
            symbols = subprocess.check_output(["readelf", "--version-info", str(path)], text=True)
            versions = {(int(a), int(b)) for a, b in re.findall(r"GLIBC_(\d+)\.(\d+)", symbols)}
            if "GLIBC_PRIVATE" in symbols or any(version > (2, 28) for version in versions):
                raise ValueError(f"{wheel.name}: {name} exceeds glibc 2.28: {sorted(versions)}")
            print(f"{wheel.name}: {name}: {arch}, maximum GLIBC {max(versions, default=(0, 0))}")
    if not libraries:
        raise ValueError(f"No native library found in {wheel.name}")


if __name__ == "__main__":
    wheels = sorted({Path(path) for pattern in sys.argv[1:] for path in glob.glob(pattern)})
    if not wheels:
        sys.exit("No wheels found to validate")
    for wheel in wheels:
        check(wheel)
