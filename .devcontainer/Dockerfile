ARG TARGETARCH
FROM quay.io/pypa/manylinux_2_28_x86_64@sha256:39df0042d5cc900b085aa25a0659368b42a0006c54c474299b785b44c1b4ff82 AS base-amd64
FROM quay.io/pypa/manylinux_2_28_aarch64@sha256:f5f03dedf61b47d69d0742d36047db78adeee3bdb7a9910ff52304750ecf2514 AS base-arm64

# AlmaLinux 8 + GCC 14, glibc 2.28 and Python 3.12.15, pinned by image digest.
# Build natively on each architecture; newer distro toolchains would raise the
# runtime glibc floor even if the wheel were labelled manylinux_2_30.
FROM base-${TARGETARCH} AS toolchain
ENV PATH="/opt/python/cp312-cp312/bin:${PATH}"
COPY rpm-packages.lock /opt/build/
RUN xargs -a /opt/build/rpm-packages.lock dnf install -y --setopt=install_weak_deps=False \
    && dnf clean all
COPY downloads.sha256 install-tools.sh /opt/build/
RUN bash /opt/build/install-tools.sh
COPY azurite/package.json azurite/package-lock.json /opt/azurite/
RUN cd /opt/azurite && npm ci --omit=dev --ignore-scripts --no-audit --no-fund
ENV PATH="/opt/azurite/node_modules/.bin:${PATH}"

FROM toolchain AS native
COPY native-common.sh build-native.sh /opt/build/
RUN bash /opt/build/build-native.sh
COPY build-aws.sh /opt/build/
RUN bash /opt/build/build-aws.sh
COPY build-azure.sh /opt/build/
RUN bash /opt/build/build-azure.sh
COPY build-io.sh /opt/build/
RUN bash /opt/build/build-io.sh

FROM native AS devcontainer
# Python dependency changes do not invalidate the native SDK build cache.
COPY requirements.lock /opt/build/requirements.lock
RUN python3 -m pip install --no-cache-dir --require-hashes --only-binary=:all: -r /opt/build/requirements.lock \
    && python3 -m pip check
# Constraints also cover wheel installs without enabling pip's hash mode for
# locally built project wheels (the environment install above checks hashes).
RUN sed -n -E 's/^([A-Za-z0-9_.-]+==[^ ;]+).*/\1/p' /opt/build/requirements.lock > /opt/build/constraints.txt
ENV PIP_CONSTRAINT=/opt/build/constraints.txt
