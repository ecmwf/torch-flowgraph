<!--
SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to torch-flowgraph

Thank you for your interest in torch-flowgraph! Contributions follow ECMWF's
[External Contributions](https://github.com/ecmwf/codex/blob/main/Guidelines/External-Contributions.md)
guidelines; this document summarises them for this repository. Support is
provided on a best-efforts basis.

## Raise an issue

If you find a bug or want to suggest a feature, please
[open an issue](https://github.com/ecmwf/torch-flowgraph/issues) and include:

* a clear description of the problem or the proposed feature;
* the steps needed to reproduce the problem, ideally with a minimal model;
* the expected and actual behaviour, including the full error message;
* the versions of torch-flowgraph, PyTorch, Python and your operating system.

For anything beyond a small fix, please discuss it in an issue before starting work.

Security vulnerabilities must **not** be reported in public issues. Follow the
private reporting process in [SECURITY.md](SECURITY.md) instead.

## Developing in torch-flowgraph

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the
   repository on GitHub to your personal or organisation account.
2. Clone your fork and set up a development environment:

   ```bash
   git clone https://github.com/<your-account>/torch-flowgraph.git
   cd torch-flowgraph
   pip install -e ".[tests]"
   pre-commit install
   ```

   Rendering images additionally requires the Graphviz `dot` executable on your `PATH`.
3. Create a branch for your change. Add tests that demonstrate the new behaviour
   or prevent the regression you are fixing, and check that the test suite and
   code-quality checks pass:

   ```bash
   pytest
   pre-commit run --all-files
   ```

4. Commit your changes and push the branch to your fork.
5. Open a pull request against the `main` branch of
   [ecmwf/torch-flowgraph](https://github.com/ecmwf/torch-flowgraph), fill in
   the pull request template, and reference any related issues.
6. Keep your pull request in sync with `main` and resolve any merge conflicts.

## Code review process

Code-quality checks and tests run through GitHub Actions on every pull request.
For security reasons, maintainers must review code changes from forks before
enabling the automated tests:

* pull requests from forks are labelled `contributor` automatically;
* after an initial review, a maintainer adds the `approved-for-ci` label, which
  runs the tests on your pull request;
* pushing new commits removes the `approved-for-ci` label, so the new changes
  are reviewed and the label added again before the tests run on them.

A pull request must pass all required checks and be approved by a maintainer
before it is merged.

If you used AI tools to generate part of your contribution, please say so in the
pull request description. You remain responsible for understanding and testing
the code you submit.

## Licensing

torch-flowgraph is licensed under the [Apache License 2.0](LICENSE), and all
contributions must comply with ECMWF's
[Copyright and Licensing](https://github.com/ecmwf/codex/blob/main/Legal/Copyright-And-Licensing.md)
policy. New source files must start with the SPDX header used by the existing
files (checked by `reuse lint` in pre-commit and CI):

```python
# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0
```

By opening a pull request you agree to the terms of the ECMWF
[Contributor Licence Agreement](https://github.com/ecmwf/codex/blob/main/Legal/Contributor-License-Agreement.md).
External contributors are welcome to add themselves to [CONTRIBUTORS](CONTRIBUTORS)
in their pull request.
