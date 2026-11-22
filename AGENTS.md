# Repository instructions

## Making changes

- Keep changes focused on the requested task. Avoid unrelated refactors and dependency upgrades.
- Preserve public Python, C, and C++ APIs, package names, and release artifact names unless the task explicitly requires changing them.
- Follow [CONTRIBUTING.md](CONTRIBUTING.md) for build, test, SPDX header, and pull request conventions.
- Run checks relevant to the changed components. Report what passed and which checks could not run, including the reason.
- Write release notes with [docs/writing-release-notes.md](docs/writing-release-notes.md). It sets the structure, the order, how to verify each claim against the tag, and the terms to use.

## Code comments

Full rules: [Code Comments](CONTRIBUTING.md#code-comments). The short version:

- Default to no comment. Add one only for a constraint the code cannot show (kernel or filesystem quirk, ownership or lifetime rule, ordering requirement, error-code mapping). Never restate what the code does.
- Keep it to one line unless the constraint truly needs more. Do not copy the density of nearby code: much of the existing code is more heavily commented than new code should be.
- State the current behavior only. Never write change history ("used to", "no longer", "previously", "now"); that belongs in the commit message or PR.
- No benchmark numbers or review debate in code. Put measurements in `docs/`.
- When you edit code, update or delete the comments around it that no longer hold.

```cpp
// Bad: restates the code, and narrates history.
// Loop over the chunks. This used to use a vector.
for (auto& chunk : chunks) { ... }

// Good: a constraint the code cannot show.
// Must complete before runai_end returns: the workers still write into these slots.
```

## Commit sign-off

- Follow the DCO requirements in [CONTRIBUTING.md](CONTRIBUTING.md) and read the [DCO](DCO) before signing off.
- Always use `git commit --signoff` (`-s`) when creating or amending commits in this repository, using the configured Git identity.
- Preserve existing authorship and other contributors' sign-offs. Do not invent identities or sign off on behalf of other contributors.
- Before pushing, verify that every commit introduced by the branch includes the contributor's `Signed-off-by:` trailer.
