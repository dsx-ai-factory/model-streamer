# Third-party notices

The native libraries (`libstreamer.so`, `libstreamers3.so`, `libstreamergcs.so`,
`libstreamerazure.so`) statically link third-party code. Apache-2.0 and the other licenses
require us to ship each component's copyright and license text with the binaries. This directory
holds those notices and the tooling to regenerate them.

## Layout

| Path | What it is |
|---|---|
| `components.json` | Manifest, the source of truth: one entry per third-party component (name, version, SPDX license, copyright, linkage, which libraries use it, upstream license/NOTICE file URLs). |
| `licenses/<id>/` | Upstream license and NOTICE files, downloaded verbatim from the pinned version. Committed. |
| `licenses/sources.json` | The URL, archive member and SHA-256 of each downloaded file. `fetch` re-downloads a file whose URL or member in `components.json` changed. `check` fails if a file is missing, was edited by hand or does not match the manifest. |
| `generate.py` | `fetch` downloads the files, `build` writes the outputs, `check` fails if the outputs or the downloaded files are stale. |
| `list_linked_components.py` | Reads linker maps and lists what was really linked. |
| `libstreamer.txt`, `libstreamers3.txt`, `libstreamergcs.txt`, `libstreamerazure.txt` | Generated. One notice per native library. |
| `sdk.txt` | Generated. The union of the four, for the C/C++ SDK archive. |

## Where each deliverable gets its notice

| Deliverable | LICENSE | Notice |
|---|---|---|
| Wheel `runai-model-streamer` | `py/runai_model_streamer/LICENSE` | `py/runai_model_streamer/THIRD_PARTY_NOTICES.txt` -> `libstreamer.txt` |
| Wheel `runai-model-streamer-s3` | `py/runai_model_streamer_s3/LICENSE` | `.../THIRD_PARTY_NOTICES.txt` -> `libstreamers3.txt` |
| Wheel `runai-model-streamer-gcs` | `py/runai_model_streamer_gcs/LICENSE` | `.../THIRD_PARTY_NOTICES.txt` -> `libstreamergcs.txt` |
| Wheel `runai-model-streamer-azure` | `py/runai_model_streamer_azure/LICENSE` | `.../THIRD_PARTY_NOTICES.txt` -> `libstreamerazure.txt` |
| C/C++ SDK archive (all four `.so`) | root `LICENSE` | `third_party_notices/sdk.txt`, packaged as `THIRD_PARTY_NOTICES.txt` at the archive root |

The wheel files are symlinks, and each `setup.py` lists `THIRD_PARTY_NOTICES.txt` in
`license_files`, so it lands in `<wheel>.dist-info/`. The release job
(`.github/scripts/release_bundle.py create`) adds `sdk.txt` to each SDK archive as
`THIRD_PARTY_NOTICES.txt`, and `verify` fails if an archive lacks it or the `LICENSE`.
The two SDK archives (x86_64 and aarch64) get the same file, the union of the four libraries.

## Dependency source and rebuilding

The files in `licenses/` contain license and notice texts, not complete dependency
source. See [Native library sources and rebuilding](../docs/native-library-rebuilding.md)
for obtaining the matching source, rebuilding with modified libaio or liburing,
and the source-delivery steps required before distributing a release.

## Regenerating after a change

Run these whenever a dependency version, a build flag or the set of linked libraries changes.

1. **Use the current devcontainer**: `docker pull ghcr.io/dsx-ai-factory/model-streamer/devcontainer:latest`
   (see `.github/workflows/build-devcontainer.yml`). Prebuilt libraries live in `/opt/<arch>-*`.
2. **Get a linker map per library and architecture**, inside the container, on a copy of the repo:
   ```
   cd cpp
   bazel build streamer:libstreamer.so --config=x86_64 --linkopt=-Wl,-Map=/tmp/libstreamer.map
   bazel build s3:libstreamers3.so --config=x86_64 --linkopt=-Wl,-Map=/tmp/libstreamers3.map
   bazel build gcs:libstreamergcs.so --config=x86_64 --linkopt=-Wl,-Map=/tmp/libstreamergcs.map
   bazel build azure:libstreamerazure.so --config=x86_64 --linkopt=-Wl,-Map=/tmp/libstreamerazure.map
   # then repeat the four builds with --config=aarch64 and different map names
   python3 ../third_party_notices/list_linked_components.py /tmp/*.map
   ```
   The map lists every archive that supplied an object file. Anything linked that is not in
   `components.json` is missing from the notices. The x86_64 and aarch64 sets must be compared;
   they differ slightly (see the notes in `components.json`).

   **Build the way CI builds.** `cpp/Makefile` passes `--define USE_SYSTEM_LIBS=${USE_SYSTEM_LIBS}`
   to the s3, gcs and azure builds, and `cpp/third_party/*.BUILD` and `*.bzl` select static or
   dynamic libraries on it. Before building the maps, check how the release workflow
   (`.github/workflows/on-release.yaml` and `build.yml`) sets it, and use the same value. At the time
   of writing CI leaves it unset, which is the default (static) branch. The setting does not change
   everything: for example `-lxml2` in `azure.bzl` is outside the select, and it links the static
   `/opt/<arch>-xml2/lib/libxml2.a` because that directory holds no shared library.

   **Trust the binary, not the build files.** A comment or a README may say a library is shared when
   it is linked statically, or the reverse. For each library, check the map (`libxml2.a` supplying
   object files means static) and `readelf -d lib<name>.so | grep NEEDED` (a library listed there is
   dynamic and is not distributed by us). Both must agree with `components.json`.
3. **Resolve versions.**
   - `/opt` libraries: the pins in `.devcontainer/Dockerfile`. For aws-sdk-cpp, the `crt/` submodule
     SHAs of the pinned tag give aws-crt-cpp, aws-c-*, aws-checksums and s2n-tls versions
     (`gh api repos/aws/aws-sdk-cpp/contents/crt/aws-crt-cpp?ref=<tag>`, then the same under
     `crt/aws-crt-cpp/crt/`; map each SHA to a tag).
   - Bazel-built libraries (gcs): inside the container run
     `bazel query --output=build //external:<repo>` for each external repository seen in the map.
     In WORKSPACE mode the first definition of a repo name wins, so the resolved version can
     differ from what a dependency's own `.bzl` file says.
4. **Find code the map cannot see.**
   - Header-only code: collect the compiler dependency files
     (`find $(bazel info output_base)/execroot/__main__/bazel-out -name '*.d'`) and list the
     external repositories whose headers were included (nlohmann/json and OpenTelemetry are
     header-only here).
   - Code nested inside a component: read each component's `NOTICE`, `LICENSE` and `third_party/`
     directory (aws-sdk-cpp bundles TinyXml2 and cJSON, BoringSSL bundles fiat-crypto, gRPC bundles
     address_sorting, protobuf bundles upb and utf8_range). Add a component for each one that is
     linked.
   - Optional features: for curl check `curl-config --features` and `libcurl.pc` in
     `/opt/<arch>-curl` (currently zlib and OpenSSL only).
   - The toolchain's static runtime (libstdc++, libgcc, glibc `libc_nonshared.a`): version from
     `dpkg -l | grep -E 'libstdc\+\+|libgcc|libc6-dev'`.
5. **Edit `components.json`**, then:
   ```
   python3 third_party_notices/generate.py fetch    # downloads any new license/NOTICE file
   python3 third_party_notices/generate.py build    # rewrites the generated *.txt
   python3 third_party_notices/generate.py check    # freshness check; CI does not run it yet
   ```
   `fetch` reports files that do not exist upstream; remove those entries.
6. **Verify the deliverables**: build the wheels and confirm `dist-info/LICENSE` and
   `dist-info/THIRD_PARTY_NOTICES.txt` are present; inspect the SDK archive for the notice file.

## Rules the generator follows

- Each library gets its own notice, listing only what is linked into that library.
- Upstream license text is reproduced verbatim. An Apache-2.0 text that matches the standard text
  is written once, in Appendix A; any variant (extra sections, third-party licenses appended) is
  reproduced in full.
- An upstream `NOTICE` file is always reproduced (Apache-2.0 section 4(d)).
- Copyright strings in `components.json` are the project's standard attribution; where the license
  text itself carries the copyright (MIT, BSD, zlib, curl) the text is authoritative.
- Dual-licensed components record the option used (liburing: MIT).

## Notes

- The CUDA driver header (`cuda.h`) is compiled against only (`libcuda.so` is loaded with
  `dlopen`); its notice is included because the header requires the disclaimer to be reproduced.
