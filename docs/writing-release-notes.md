# Writing release notes

How to write the notes for a Model Streamer release, for people and for AI coding agents. The first notes written to this guide are [0.17.0](release-notes/0.17.0.md) and [0.17.1](release-notes/0.17.1.md). Use them as the model for shape and tone.

Release notes answer one question for someone who runs Model Streamer: what changed, and what do I have to do? They are not a commit log. A release's notes are written once, at release time, from the changes in that release.

## Where they live

- Write the notes in `docs/release-notes/X.Y.Z.md`. Tags have no `v` prefix. Review them in a PR to `master`, then land the same file on `release/X.Y` before tagging, so the tag contains its own notes.
- After the release workflow has published, use the same text as the GitHub release body: `gh release edit X.Y.Z --notes-file docs/release-notes/X.Y.Z.md`. The workflow keeps any notes that already exist, so the body is set by hand.
- Add the new file to the release notes line in [docs/README.md](README.md).
- Markdown files carry no license header. [CONTRIBUTING.md](../CONTRIBUTING.md) exempts them.

## Gather the facts

1. **Get the range.** List the commits between the previous tag and this one: `git log --no-merges <previous>..<tag>`. Map each commit to its PR with `gh api repos/dsx-ai-factory/model-streamer/commits/<sha>/pulls`. Many CI commits have no PR number in their title. A commit with no PR at all is listed by its commit link.
2. **Handle cherry-picks.** A patch release is cut from `release/X.Y` and its PRs are backports with new numbers. Link the PR in the tag, and add `from #N` for the original, which holds the discussion and the measurements. Credit the author of the original change, not whoever opened the backport.
3. **Check every claim against the tag, not `master`.** A variable, default or API belongs in the notes only if it is in the tagged tree: `git grep -n <name> <tag> -- <path>`. Read defaults from the code and from `docs/src/env-vars.md` at the tag, not from the PR text. The ring buffer PR says 4 buffers; the tag says 18.
4. **Check what was published.** Read the wheel filenames for each version at `https://pypi.org/pypi/runai-model-streamer/<version>/json` and the release assets. A changed `manylinux` tag is an Upgrade Note: 0.17.0 was tagged `manylinux_2_30`, 0.17.1 `manylinux_2_28`.
5. **Read the PR description.** Its test results and "not in this PR" lines say what is verified. A PR that says its documentation follows later can describe a default that changed afterwards.
6. **Say which API a change reaches.** Write Python API, C API or both. GPU destinations reached only the C API in 0.17.0, and a reader who uses vLLM would otherwise assume they apply to them.
7. **Compute the counts.** Take the number of PRs and contributors from the range you listed. A contributor is a first-time contributor only if they have no merged PR before the previous release: `gh pr list --repo dsx-ai-factory/model-streamer --author <login> --state merged --search "merged:<YYYY-MM-DD"`, with the previous release's date.

## Order by what matters most

- **Lead with the headline.** It is what users can do now that they could not before. Find it by reading what changed in `docs/src/` between the two tags: `git diff <previous> <tag> -- docs/src`. A constraint that was removed, such as a host buffer sized for the whole model, is the headline.
- **Then order by effect on the reader.** Changes every Python user gets by upgrading come first, memory and then speed. Then new interfaces such as the C/C++ SDK, then reliability and security, then packaging, then project and CI changes.
- **For a patch release, lead with what stopped working.** A wheel that does not install, or a load that fails, comes before an improvement.
- **Keep one order.** The introduction, the Summary, Features & Improvements and the Contributors lines follow the same order.
- **Upgrade Notes come before the feature lists.** They need action, so a reader should meet them first.

## Structure

Use these sections, in this order. Omit a section that would be empty, except Known Issues, which says none were recorded.

| Section | What it holds |
| --- | --- |
| Title and introduction | `# Model Streamer X.Y.Z - Release Notes`, then one paragraph |
| Summary | One `###` subsection per headline theme |
| Baseline | The table of requirements: operating system and glibc, architectures, framework, storage, packages |
| Upgrade Notes | Changes that need action or change behavior |
| Features & Improvements | `###` groups in the same order as the Summary |
| Bug Fixes | One bullet per fix users could notice |
| Testing & CI | Only changes a contributor or integrator would want to know |
| Documentation | New and changed pages |
| Known Issues | Defects, with a workaround where one exists |
| Contributors | Counts, thanks, and first-time contributors |
| Full Changelog | The compare link for the range |

## Write each part

**Introduction.** One paragraph. Open with the headline in bold, then name the next three or four themes in bold, and end with a measured result when there is one, with what it was measured on. A patch release opens by saying it is a patch release for the version before it. Do not put counts here; they belong in Contributors.

**Summary.** The heading is `### Theme: Detail & Detail`. The paragraph has the capability as its subject, takes three to five sentences, and links at most four PRs. Name a mechanism, such as `io_uring`, only where the reader must act on it.

**Features & Improvements.** One bullet per change, led by a bold feature name in Title Case, then a past-tense sentence that says what the user gets, then the PR link: `- **Ring Buffer:** Replaced the … (#167).` Use *Added*, *Fixed*, *Replaced*, *Reworked*, *Removed*. Say what changed for the user, not which function changed.

**Upgrade Notes.** Describe what changes and who is affected, then give the action as a quote that starts `> **Migrate:**`. Include every changed default, every environment variable whose meaning changed, every C API signature change, every change to what a yielded tensor is, and every changed wheel tag.

**Known Issues.** A known issue is a defect: something that fails or misbehaves today in a supported setup. Take them from the open GitHub issues. Give each one a `####` title, the symptom, the cause when it is known, and a `**Workaround:**` line when one exists. Things Model Streamer does not support by design belong in the docs.

**Contributors.** State the counts, thank everyone, and give each first-time contributor one line that says what they did.

## Words and tone

- *Model Streamer* is the product. Do not use the earlier name. Write the names that did not change exactly: `runai-model-streamer`, `runai_model_streamer` and the `RUNAI_STREAMER_*` variables.
- Use *ring buffer*, *submission*, *rank*, *node*, *file system*, *object storage*, *host memory* and *device memory*, and *Python API*, *C API* and *C/C++ SDK*.
- Write in the past tense, in plain sentences. Do not use marketing words, and do not write "we".
- Give numbers with their unit and with what they were measured on, and link to the PR that holds the data. A number measured on a PR's branch is "measured for" that change, not a property of the release.
- Do not mention customers, partners or internal organizations or tools. Name an engine such as vLLM or SGLang only where the reader must act on it or the number was measured on it.

## Links

In a Markdown file in the repository, `#123` and `@name` do not turn into links. Write each PR as `[#123](https://github.com/dsx-ai-factory/model-streamer/pull/123)`, each issue with `/issues/123`, and each person as `[@name](https://github.com/name)`. The same text then works in the file and in the GitHub release body. Link documents by their address at the tag (`blob/X.Y.Z/…`) when the page must match the release.

## Before you publish

- Every number is traced to a source, and every variable, default and API is in the tag.
- Every default was read from the code at the tag, not from a PR.
- The wheel tags match the published filenames.
- Each change says which API it reaches.
- Nothing that is a limitation by design is in Known Issues.
- Upgrade Notes cover every changed default, variable, signature and wheel tag.
- Every link resolves, and the counts match the range.
- The order follows the headline, then user impact.
