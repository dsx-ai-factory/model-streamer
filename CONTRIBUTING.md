# Contributing to Model Streamer

Thank you for your interest in contributing to Model Streamer! This document provides guidelines and instructions to help you get started with contributing to our project.

## Getting Started
### New Contributors
We're excited to help you make your first contribution! Whether you're looking to file issues, develop features, fix bugs, or improve documentation, we're here to support you through the process.

Browse issues labeled `good first issue` or `help wanted` on GitHub for an easy introduction.

### Developers
The main building blocks of Model Streamer are documented in the [docs](docs/README.md) folder. Here are the key components:
- `cpp/streamer` - Core C++ streaming engine
- `cpp/s3`, `cpp/gcs`, `cpp/azure` - Object storage backend clients
- `cpp/cc` - C API exposed to Python via ctypes
- `py/runai_model_streamer` - Python SDK
- `py/runai_model_streamer_s3`, `py/runai_model_streamer_gcs`, `py/runai_model_streamer_azure` - Storage-specific Python packages

We recommend reviewing the [README](README.md) to understand the system architecture and build process before making significant contributions.

## How to Contribute
### Reporting Issues
Open an issue with a clear description, steps to reproduce, and relevant environment details.

### Improving Documentation
Help us keep the docs clear and useful by fixing typos, updating outdated information, or adding examples.

### Contributing Changes
- Fork and Clone – Begin by forking the repository and cloning it to your local machine.
- Create a Branch – Use a descriptive branch name, such as `feature/add-cool-feature` or `bugfix/fix-issue123`.
- Make Changes – Keep your commits small, focused, and well-documented. For build and test instructions, refer to the [Development section](README.md#development) of the README.
- Submit a PR – Open a pull request and reference any relevant issues or discussions.
- Coverage - Please look at the coverage change details and create unit tests, integration tests or end-to-end tests to cover new functionality or changes.

### PR Title Guidelines

We recommend following the [Conventional Commits](https://www.conventionalcommits.org/) title specification. The format is:

```
<type>[optional scope]: <description>
```

#### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that don't affect code meaning (formatting, whitespace)
- **refactor**: Code changes that neither fix a bug nor add a feature
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **build**: Changes to build system or dependencies
- **ci**: Changes to CI/CD configuration
- **chore**: Other changes that don't modify src or test files
- **revert**: Reverts a previous commit

#### Scopes (Optional)

Common scopes for Model Streamer:
- `streamer`
- `s3`
- `gcs`
- `azure`
- `py`
- `cpp`
- `ci`
- `docs`

#### Breaking Changes

Breaking changes MUST be indicated by adding `!` after the type/scope: `feat(s3)!: remove deprecated field`

#### Examples

```
feat(s3): add retry with bounded deadline for transient chunk failures
fix(azure): resolve race condition in credential refresh
docs: update installation guide
refactor(streamer): simplify ring buffer allocation
feat(gcs)!: remove deprecated field from client config
```

#### Tips

- Use the imperative mood: "add feature" not "added feature"
- Don't end with a period

### Pull Request Checklist

Before introducing major changes, we strongly recommend opening a PR that outlines your proposed design.
Each pull request should meet the following requirements:
- All tests pass – Run the full test suite locally with: `make test`
- Test coverage – Add or update tests for any affected code.
- Documentation – Update relevant documentation to reflect your changes.
- PR description – Clearly describe what changed and why.

## Getting Help
Need support or have a question? We're here to help:
- Report issues or ask questions by [opening an issue on GitHub](https://github.com/dsx-ai-factory/model-streamer/issues).

## License
By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

Thank you for your interest and happy coding!
