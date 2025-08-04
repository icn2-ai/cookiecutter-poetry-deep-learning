# Cookiecutter Poetry Deep Learning (ICN2)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/icn2-ai/cookiecutter-poetry-deep-learning)
---

This is a Cookiecutter template for deep-learning Python projects. It has all the necessary tools for development, testing, and deployment: 

- [Poetry](https://python-poetry.org/) for dependency management
- CI/CD with [GitHub Actions](https://github.com/features/actions)
- Pre-commit hooks with [pre-commit](https://pre-commit.com/)
- Code quality with [ruff](https://github.com/charliermarsh/ruff), [mypy](https://mypy.readthedocs.io/en/stable/) or [pyright](https://github.com/microsoft/pyright), [deptry](https://github.com/fpgmaas/deptry/) and [prettier](https://prettier.io/)
- Publishing to [PyPI](https://pypi.org) or [Artifactory](https://jfrog.com/artifactory) by creating a new release on GitHub
- Testing and coverage with [pytest](https://docs.pytest.org/en/7.1.x/) and [codecov](https://about.codecov.io/)
- Documentation with [MkDocs](https://www.mkdocs.org/)
- Compatibility testing for multiple versions of Python with [Tox](https://tox.wiki/en/latest/)
- Containerization with [Docker](https://www.docker.com/)
- Development environment with [VSCode devcontainers](https://code.visualstudio.com/docs/devcontainers/containers)

---

<!--
<p align="center">
  <a href="https://fpgmaas.github.io/cookiecutter-poetry/">Documentation</a> - <a href="https://github.com/fpgmaas/cookiecutter-poetry-example">Example</a> -
  <a href="https://pypi.org/project/cookiecutter-poetry/">PyPI</a>
</p>
-->

---

## Generated projects architecture

The generated projects with this template will have the following architecture:

```
project_name
├── configs             # Configuration files for experiments
├── data                # Data for training and testing
├── docs                # Documentation files
├── notebooks           # Jupyter notebooks
├── tests               # Unit tests for the project
├── weights             # Trained models
├── project_slug (src)  # Code for the project 
    ├── base
    ├── data            # Data processing and loading
    ├── logger          # Logging
    ├── model           # Model definition, metrics, and losses
    ├── trainer         # Trainer class
    ├── utils           # Utility functions
    ├── visualization   # Visualization functions
    ├── test.py         
    ├── train.py        
├── Makefile
├── README.md
```

Files for version control, documentation, and CI/CD are not shown in the tree above.

## Quickstart

On your local machine, install `cookiecutter` and directly pass the URL to this
Github repository to the `cookiecutter` command:

```bash
pip install cookiecutter
cookiecutter https://github.com/icn2-ai/cookiecutter-poetry-deep-learning.git
```

If you have not installed poetry, follow the instructions [here](https://python-poetry.org/docs/#installation).

Create a repository on GitHub, and then run the following commands, replacing `<project-name>`, with the name that you gave the Github repository and
`<github_author_handle>` with your Github username.

```bash
cd <project_name>
git init -b main
git add .
git commit -m "Init commit"
git remote add origin git@github.com:<github_author_handle>/<project_name>.git
git push -u origin main
```

Install the environment and the pre-commit hooks with:

```bash
make install
```

Then, before making the push to the remote repository, run the pre-commit hooks with:

```bash
make check
```

It might give an error the first time you run it, but `mypy` and `ruff` will correct the code style and type hints. You can run the pre-commit hooks again to check if everything is correct.
After that, you can do the push. 

You are now ready to start development on your project! The CI/CD
pipeline will be triggered when you open a pull request, merge to main,
or when you create a new release.


<!-- 
To finalize the set-up for publishing to PyPI or Artifactory, see
[here](https://fpgmaas.github.io/cookiecutter-poetry/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see
[here](https://fpgmaas.github.io/cookiecutter-poetry/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/codecov/).
--> 

## Acknowledgements

This project is based on [Florian Maas'](https://github.com/fpgmaas) repository,
[cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry), for the development of Python projects with 
Poetry, including CI/CD and code quality tools. The generated projects' architecture is inspired by 
[Victor Huang's](https://github.com/victoresque) template fo deep-learning projects, 
[pytorch-template](https://github.com/victoresque/pytorch-template).

