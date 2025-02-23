# AutoRA Experimentalist Latin Hypercube Sampling

## Quickstart Guide

### Link to GitHub Repository

You should link the root folder of this project to an existing repository on GitHub. 

1. First, initialize your local project folder with `git init`.
2. If you don't have a remote repository yet, you should create one on GitHub (a sensible name for the repository would be autora-experimentalist-lhs).  If you already have a remote repository on Github, you can jump to Step 3. 
3. Link the local project folder to the remote repository via `git remote add origin https://github.com/OWNER/REPOSITORY.git`. See [this guide](https://docs.github.com/en/get-started/getting-started-with-git/managing-remote-repositories) for more instructions.
4. To ensure that the remote repository has been added correctly, you can run `git remote -v `from within the project folder.

### Virtual Environment 
Install this in an environment using your chosen package manager. In this example we are using virtualenv

Install:
- python (3.8 or greater): https://www.python.org/downloads/
- virtualenv: https://virtualenv.pypa.io/en/latest/installation.html

Create a new virtual environment:
```shell
virtualenv venv
```
*Note: You want to ensure that the python version matches that of autora. If necessary 
you can specify the respective python version directly, e.g., ``virtualenv venv --python=python3.9``*

Activate it:
```shell
source venv/bin/activate
```

### Install Dev Dependencies

Use `pip install` to install the current project (`"."`) in editable mode (`-e`) with dev-dependencies (`[dev]`):
```shell
pip install -e ".[dev]"
```

*Note: You may install new dependencies via ``pip install packagename`` inside your virtual environment. If those
dependencies are vital to your package, you will have to add them to the ``pyproject.toml`` (see Step 6 of the 
Contribution Guide).*

## Contribution Guide

### Experimentalist
A method that identifies novel experiment conditions $X'$ that yield scientific merit.
*Example: The [Novelty Experimentalist](https://github.com/AutoResearch/autora-experimentalist-novelty) selects novel experiment 
conditions $X'$ with respect to a pairwise distance metric applied to existing experiment conditions $X$.* <br>

More information on how to implement an AutoRA experimentalist component can be found in the respective [contributor documentation](https://autoresearch.github.io/autora/contribute/modules/experimentalist/).
### Step 1: Implement Your Code

You may now add your code to `src/autora/experimentalist/lhs/__init__.py` file. You may 
also add additional files in this folder. Just make sure to import the core function or class of your feature
in the `__init__.py` if it is implemented elsewhere. 

### Step 2 (Optional): Add Tests

It is highly encouraged to add unit tests to ensure your code is working as intended. These can be [doctests](https://docs.python.org/3/library/doctest.html) or test cases in `tests/test_lhs.py`.

*Note: Tests are required if you wish that your feature becomes part of the main 
[autora](https://github.com/AutoResearch/autora) package. However, regardless of whether you choose to implement tests, 
you will still be able to install your package separately, in addition to autora.* 

### Step 3 (Optional): Add Documentation

It is highly encouraged that you add documentation of your package in your `docs/index.md`. You can also add new pages 
in the `docs` folder. Update the `mkdocs.yml` file to reflect structure of the documentation. For example, you can add 
new pages or delete pages that you deleted from the `docs` folder.

*Note: Documentation is required if you wish that your feature becomes part of the main 
[autora](https://github.com/AutoResearch/autora) package. However, regardless of whether you choose to write
documentation, you will still be able to install your package separately, in addition to autora.*

### Step 4: Add Dependencies

In pyproject.toml add the new dependencies under `dependencies`

Install the added dependencies
```shell
pip install -e ".[dev]"
```

### Step 5: Publish Your Package

Once your project is implemented, you may publish it as subpackage of AutoRA. If you have not thoroughly vetted your project or would otherwise like to refine it further, you may 
nervous about the state of your package–you will be able to publish it as a pre-release, indicating to users that
the package is still in progress.

#### Step 5.1: Update Metadata

To begin publishing your package, update the metadata under `project` in the pyproject.toml file to include 
- name
- description
- author-name
- author-email
Also, update the URL for the repository under `project.urls`.

#### Step 5.2 Publish via GitHub Actions

To automate the publishing process for your package, you can use a GitHub action:
- Add a github secret in your repository named PYPI_API_TOKEN, that contains a PyPI token of your account
- Add the GitHub action to the `.github/workflows` directory: For example, you can use the default publishing action:
  - Navigate to the `actions` on the GitHub website of your repository.
  - Search for the `Publish Python Package` action and add it to your project
- Create a new release: Click on `create new release` on the GitHub website of your repository.
- Choose a tag (this is the version number of the release. If you didn't set up dynamic versioning it should match the version in the `pyproject.toml` file)
- Generate release notes automatically by clicking `generate release`, which adds the markdown of the merged pull requests and the contributors.
- If this is a pre-release check the box `set as pre-release`
- Click on `publish release`
### Step 6: Add the package to [`autora`](https://github.com/autoresearch/autora)
Once your package is working and published, you can **make a pull request** on 
[`autora`](https://github.com/autoresearch/autora) to have it vetted and added to the "parent" package.
To do so, you'll need to clone the parent repository, add your package to it as an optional dependency, and make sure your documentation is imported.
Then create a pull request with the changes to let us know about your contribution.

#### Add the package as optional dependency 
In the `pyproject.toml` file of your cloned [`autora`](https://github.com/autoresearch/autora) package, 
add an optional dependency for your new package in the `[project.optional-dependencies]` section:

```toml
example-contribution = ["autora-example-contribution==1.0.0"]
```

!!! success
    Ensure you include the version number.

Add the example-contribution to be part of the corresponding all-contribution-type dependencies:
```toml
all-contribution-type = [
    ...
    "autora[example-contribution]",
    ...
]
```

### Add documentation from the package repository
Import your documentation in the `mkdocs.yml` of the [`autora`](https://github.com/autoresearch/autora) package:
```yml
- User Guide:
  - Contribution Type:
    - Overview: 'contribution-type/overview.md'
    ...
    - Example Contribution: '!import https://github.com/example-contributor/example-contribution/?branch=v1.0.0&extra_imports=["mkdocs/base.yml"]'
    ...
```

## Questions & Help

If you have any questions or require any help, please add your question in the 
[Contributor Q&A of AutoRA Discussions](https://github.com/orgs/AutoResearch/discussions/categories/contributor-q-a).
We look forward to hearing from you!
