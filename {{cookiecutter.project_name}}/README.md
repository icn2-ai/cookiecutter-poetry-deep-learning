# {{cookiecutter.project_name}}

{{cookiecutter.project_description}}

- **Github repository**: <https://github.com/{{cookiecutter.author_github_handle}}/{{cookiecutter.project_name}}/>
<!-- - **Documentation** <https://{{cookiecutter.author_github_handle}}.github.io/{{cookiecutter.project_name}}/>-->

## Getting started with your project

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin https://github.com/{{cookiecutter.author_github_handle}}/{{cookiecutter.project_name}}.git
git push -u origin main
```

If this step has already been completed (you already have a github repository), you can modify this README file to present your project and add the following command to clone it: :

```bash
git clone https://github.com/{{cookiecutter.author_github_handle}}/{{cookiecutter.project_name}}.git
````

To manage the dependencies and the environment, we use Poetry. If you have not installed Poetry, follow the instructions 
[here](https://python-poetry.org/docs/#installation). You will need, at least, **version 2.1.1 of Poetry**. Once poetry is installed, install the environment and the pre-commit hooks with

```bash
make install
```

You are now ready to start training your new AI model!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

<!-- To finalize the set-up for publishing to PyPI or Artifactory, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-poetry/features/codecov/).
-->
---

Repository initiated with [ai-icn2/cookiecutter-poetry-deep-learning](https://github.com/ai-icn2/cookiecutter-poetry-deep-learning).
