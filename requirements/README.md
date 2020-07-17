## Detailed Install Instructions

### Local environment

To setup a local development environment for the project, simply create a new Conda environment using
[`neuralteleportation.yml`](neuralteleportation.yml). Creating a new Conda environment from a file can
be done using the following command (from the root folder of the project):
```bash
conda env create -f requirements/neuralteleportation.yml
conda activate neuralteleportation
```
Unless you manually edit the first line (`name: neuralteleportation`) of the file, the environment will be named
`neuralteleportation` by default.

Do not forget to also install the project's package, which will allow to import modules from the project, using the
following command:
```bash
pip install -e .
```

### Compute Canada environment

Here is a brief step by step description of how to setup a working environment for the project on Compute Canada's
servers. For more information on how to configure virtual environments on Compute Canada's servers,
please refer to their own
[documentation on the subject](https://docs.computecanada.ca/wiki/Python#Creating_and_using_a_virtual_environment).

On Compute Canada's servers, the recommended tool to manage virtual environments is Python's own virtualenv, rather
than Conda. It is also recommended to use packages compiled specially by Compute Canada for the servers' architectures,
instead of generic packages automatically downloaded by pip. This last recommendation explains why some of the
following commands add options that are not generally seen when configuring local virtual environments.

When using virtualenv, it is necessary to first create an environment like below. Note that it is important to first
load the Python module to ensure that the virtual environment's base Python version is the appropriate one.
```bash
module load python/3.7
module load scipy-stack # For scipy, matplotlib and pandas
virtualenv --no-download <path_to_virtual_env>
```

After the virtual environment is created, it is necessary to activate it and update the base environment.
```bash
source <path_to_virtual_env>/bin/activate
pip install --upgrade setuptools pip wheel
```

Afterwards, the environment's packages can be installed from the requirements file, like follows:
```bash
pip install --no-index -r requirements/computecanada_wheel.txt
pip install -r requirements/computecanada_no_wheel.txt
```

Do not forget to also install the project's package, which will allow to import modules from the project, using the
following command:
```bash
pip install -e .
```