# Repository instructions

## Making changes

- Keep changes focused on the requested task. Avoid unrelated refactors and dependency upgrades.
- Preserve public Python, C, and C++ APIs, package names, and release artifact names unless the task explicitly requires changing them.
- Follow [CONTRIBUTING.md](CONTRIBUTING.md) for build, test, SPDX header, and pull request conventions.
- Run checks relevant to the changed components. Report what passed and which checks could not run, including the reason.
- Write release notes with [docs/writing-release-notes.md](docs/writing-release-notes.md). It sets the structure, the order, how to verify each claim against the tag, and the terms to use.

## Commit sign-off

- Follow the DCO requirements in [CONTRIBUTING.md](CONTRIBUTING.md) and read the [DCO](DCO) before signing off.
- Always use `git commit --signoff` (`-s`) when creating or amending commits in this repository, using the configured Git identity.
- Preserve existing authorship and other contributors' sign-offs. Do not invent identities or sign off on behalf of other contributors.
- Before pushing, verify that every commit introduced by the branch includes the contributor's `Signed-off-by:` trailer.
