# 🎲 Contributing to SDG-PGMs

Thank you for your interest in contributing to SDG-PGMs!

We welcome contributions from the community and sincerely appreciate your efforts to improve the project. Whether you're fixing a typo, reporting a bug, proposing a new feature, or implementing a major enhancement, your work helps make SDG-PGMs better for everyone.

This guide will help you get started with the contribution process.

## Table of Contents

- [Getting Started](#getting-started)
- [Ways to Contribute](#ways-to-contribute)
- [Feature Requests](#feature-requests)
- [Development Guide](#development-guide)
- [Submitting Changes](#submitting-changes)
- [Code of Conduct](#code-of-conduct)
- [Signing off on your work](#signing-off-on-your-work)


## Getting Started

Welcome to the SDG-PGMs community! We're excited to have you here.

Whether you're new to the project or ready to dive in, the resources below will help you get oriented and productive quickly:

1. **[README.md](https://github.com/NVIDIA-NeMo/SDG-PGMs/blob/main/README.md)** – best place to start to learn the basics of the project

2. **[Data Designer plugin](https://github.com/NVIDIA-NeMo/SDG-PGMs/blob/main/src/data_designer_plugins/)** – integration plugin for using SDG-PGMs as a column generator in Data Designer

3. **[NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner)** – compound-AI framework for synthetic data generation

## Ways to Contribute

There are many ways to contribute to SDG-PGMs:

### Bug Fixes

Found a bug? Before reporting, please
1. Verify you're using the latest code from the `main` branch
2. Search for duplicates in the [issue tracker](https://github.com/NVIDIA-NeMo/SDG-PGMs/issues)

When [creating a bug report](https://github.com/NVIDIA-NeMo/SDG-PGMs/issues/new), please include:
- Git commit hash or branch
- Python version and operating system
- Minimal reproducible example
- Expected vs. actual behavior
- Full error messages and stack traces

If you are interested in fixing the bug yourself, that's great! Please follow the [development guide](#development-guide) to get started.

### Feature Implementation
Want to add new functionality? Please open a [feature request](#feature-requests) to discuss the idea and get feedback before investing significant time on the implementation.

### Documentation Improvements
Documentation is crucial for user adoption. Contributions that clarify usage, add examples, or fix typos are highly valued.

### Examples and Tutorials
Share your use cases! Example notebooks and tutorials help others understand how to leverage SDG-PGMs effectively.

### Test Coverage
Help us improve test coverage by adding tests for untested code paths or edge cases.

## Feature Requests
SDG-PGMs is designed to be flexible and extensible, and we welcome your ideas for pushing its capabilities even further! To keep the library maintainable while supporting innovation, we take an incremental approach when adding new features:

### How We Grow SDG-PGMs
1. **Explore what's possible**: Can your use case be achieved with the current PGM building blocks? SDG-PGMs is designed to be composable — sometimes creative combinations of existing cascaded PGMs and post-processing steps can accomplish what you need. Check out our examples or open an issue if you'd like help exploring this!

2. **Extend through new PGM components**: If existing features aren't quite enough, consider implementing your idea as a new PGM stage, post-processing step, or sampling strategy that extends the framework.

3. **Integrate into the core library**: If your component proves broadly useful and aligns with SDG-PGMs' goals, we'd love to integrate it into the core library! We're happy to discuss whether it's a good fit and how to move forward together.

This approach helps us grow thoughtfully while keeping SDG-PGMs focused and maintainable.

### Submitting a Feature Request
Open a [new issue](https://github.com/NVIDIA-NeMo/SDG-PGMs/issues/new) with:

- **Clear title**: Concise description of the feature
- **Use case**: Explain what problem this solves and why it's important
- **Proposed solution**: Describe how you envision the feature working
- **Alternatives considered**: Other approaches you've thought about
- **Examples**: Code examples or mockups of how users would interact with the feature
- **Willingness to implement**: Are you interested in implementing this yourself?

## Development Guide

### Initial Setup
0. **Create or find an issue**

    Before starting work, ensure there's an issue tracking your contribution:

    - For bug fixes: Search [existing issues](https://github.com/NVIDIA-NeMo/SDG-PGMs/issues) or [create a new one](https://github.com/NVIDIA-NeMo/SDG-PGMs/issues/new)
    - For new features: Open a [feature request](#feature-requests) to discuss the approach first
    - Comment on the issue to let maintainers know you're working on it

1. **Fork and clone the repository**

    Start by [forking the SDG-PGMs repository](https://github.com/NVIDIA-NeMo/SDG-PGMs/fork), then clone your fork and add the upstream remote:

    ```bash
    git clone https://github.com/YOUR_GITHUB_USERNAME/SDG-PGMs.git

    cd SDG-PGMs

    git remote add upstream https://github.com/NVIDIA-NeMo/SDG-PGMs.git
    ```

2. **Install dependencies**

    ```bash
    uv sync
    ```

3. **Verify your setup**

    ```bash
    uv run pytest
    ```

    If no errors are reported, you're ready to develop.

### Making Changes

1. **Create a feature branch**

    ```bash
    git checkout main
    git pull upstream main
    git checkout -b <username>/<type-of-change>/<issue-number>-<short-description>
    ```

    Example types of change:

    - `feat` for new features
    - `fix` for bug fixes
    - `docs` for documentation updates
    - `test` for testing changes
    - `refactor` for code refactoring
    - `chore` for chore tasks
    - `style` for style changes
    - `perf` for performance improvements

    Example branch name:

    - `jdoe/feat/42-add-custom-sampling-strategy` for a new feature by @jdoe, addressing issue #42

2. **Develop your changes**

    Please follow the patterns and conventions used throughout the codebase.

3. **Test and validate**

    ```bash
    uv run ruff check --fix .   # Fix linting issues
    uv run ruff format .        # Format code
    uv run pytest               # Run all tests
    ```

    **Writing tests**: Place tests in the `tests/` directory mirroring the source structure. Mock external services with `unittest.mock`, and test both success and failure cases.

4. **Commit your work**

    Write clear, descriptive commit messages, optionally including a brief summary (50 characters or less) and reference issue numbers when applicable (e.g., "Fixes #42").

    ```bash
    git commit -m "Add custom sampling strategy for cascaded PGMs" -m "Fixes #42"
    ```

5. **Stay up to date**

    Regularly sync your branch with upstream changes:

    ```bash
    git fetch upstream
    git merge upstream/main
    ```

## Submitting Changes

### Before Submitting

Ensure your changes meet the following criteria:

- All tests pass (`uv run pytest`)
- Code is formatted and linted (`uv run ruff check . && uv run ruff format --check .`)
- New functionality includes tests
- Documentation is updated (README, docstrings, examples)
- License headers are present on all new files
- Commit messages are clear and descriptive

### Creating a Pull Request

1. **Push your changes** to your fork:

    ```bash
    git push origin <username>/<type-of-change>/<issue-number>-<short-description>
    ```

2. **Open a pull request** on GitHub from your fork to the main repository

3. **Respond to review feedback** and update your PR as needed

### Pull Request Review Process

- Maintainers will review your PR and may request changes
- Address feedback by pushing additional commits to your branch
- Reply to the feedback comment with a link to the commit that addresses it
- Once approved, a maintainer will merge your PR
- Your contribution will be merged into the main branch!

## Code of Conduct
SDG-PGMs is committed to providing a welcoming and inclusive environment for all contributors.

### License File Headers
All code files that are added to this repository must include the appropriate NVIDIA copyright header:

```python
# SPDX-FileCopyrightText: Copyright (c) {YEAR} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

## Signing off on your work

When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project license. All contributors are asked to sign the SDG-PGMs [Developer Certificate of Origin (DCO)](https://github.com/NVIDIA-NeMo/SDG-PGMs/blob/main/DCO) when submitting their first pull request. The process is automated by a bot that will comment on the pull request. Our DCO is the same as the Linux Foundation requires its contributors to sign.

---

Thank you for contributing to SDG-PGMs! Your efforts help make probabilistic graphical models for synthetic data generation more accessible and powerful for everyone.
