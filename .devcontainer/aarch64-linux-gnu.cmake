# /opt/toolchains/aarch64-linux-gnu.cmake
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR aarch64)

# Compiler and linker paths
SET(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
SET(CMAKE_LINKER /usr/bin/aarch64-linux-gnu-ld)
SET(CMAKE_AR /usr/bin/aarch64-linux-gnu-ar)
SET(CMAKE_NM /usr/bin/aarch64-linux-gnu-nm)
SET(CMAKE_RANLIB /usr/bin/aarch64-linux-gnu-ranlib)

# Specify sysroot if needed
SET(CMAKE_FIND_ROOT_PATH /opt/aarch64-sysroot)
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
