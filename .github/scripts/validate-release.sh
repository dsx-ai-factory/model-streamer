#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

version=${GITHUB_REF_NAME:?Missing release tag}
if [[ ${GITHUB_REF_TYPE:-} != tag || ! "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
    echo '::error::Release tags must have the form X.Y.Z (no v prefix or prerelease suffix).'
    exit 1
fi
branch="release/${version%.*}"
git fetch --no-tags origin "+refs/heads/$branch:refs/remotes/origin/$branch"
commit=$(git rev-parse "refs/tags/$version^{commit}")
if [[ "$commit" != "$(git rev-parse HEAD)" ]] ||
   ! git merge-base --is-ancestor "$commit" "refs/remotes/origin/$branch"; then
    echo "::error::Tag $version must point to the checked-out commit on $branch."
    exit 1
fi
printf 'version=%s\nbranch=%s\ncommit=%s\n' "$version" "$branch" "$commit" >> "$GITHUB_OUTPUT"
