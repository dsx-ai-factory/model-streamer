# Read a file with the native SDK

This C99 example reads and verifies the first 4096 bytes (or the entire file if
shorter) of a local file. It uses buffered I/O so it needs neither aligned buffers
nor io_uring. Empty files work too. The same source can be compiled as C++17.

Unpack the SDK and set `SDK_PREFIX` to its absolute path:

```sh
export SDK_PREFIX="$HOME/runai-sdk"
mkdir -p "$SDK_PREFIX"
tar -xzf runai-model-streamer-cpp-sdk-<version>-linux-<arch>.tar.gz \
  -C "$SDK_PREFIX" --strip-components=1
```

With CMake:

```sh
cmake -S "$SDK_PREFIX/examples/hello_streamer" -B build-hello \
  -DCMAKE_PREFIX_PATH="$SDK_PREFIX"
cmake --build build-hello
./build-hello/hello_streamer /path/to/local/file
```

CMake supplies the library search path for this build. With pkg-config, set the
metadata path and explicitly embed a runtime search path:

```sh
export PKG_CONFIG_PATH="$SDK_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
cc -std=c99 -Wall -Wextra -Werror \
  "$SDK_PREFIX/examples/hello_streamer/main.c" \
  $(pkg-config --cflags --libs runai-streamer) \
  -Wl,-rpath,"$SDK_PREFIX/lib" -o hello_streamer
./hello_streamer /path/to/local/file
```

Use `c++ -x c++ -std=c++17` in place of `cc -std=c99` to build as C++.
`pkg-config --modversion runai-streamer` reports the SDK version.

Both discovery files find headers and libraries relative to their own location.
If you move the SDK, configure/build your application again with the new prefix.
Keep the cloud plugins beside `libstreamer.so`. This filesystem example needs
only the core library; cloud plugins are loaded on demand.
