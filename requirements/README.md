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

> NOTE: The job submission scripts handle automatically handle the creation of virtual environments directly on the
> compute nodes, which is arguably the optimal way to manage environments. Therefore, the instructions detailed below
> do not need to be run, but rather serve as reference in case somebody would want to setup an environment accessible
> from login nodes.  

On Compute Canada's servers, the recommended tool to manage virtual environments is Python's own virtualenv, rather
than Conda. It is also recommended to use packages compiled specially by Compute Canada for the servers' architectures,
instead of generic packages automatically downloaded by pip. This last recommendation explains why some of the
following commands add options that are not generally seen when configuring local virtual environments.

When using virtualenv, it is necessary to first create an environment like below. Note that it is important to first
load the Python module to ensure that the virtual environment's base Python version is the appropriate one.
```bash
module load python/3.7
virtualenv --no-download <path_to_virtual_env>
```
After the virtual environment is created, it is necessary to activate it.
```bash
source <path_to_virtual_env>/bin/activate
```
Afterwards, the environment's packages can be installed from the requirements file, like follows:
```bash
pip install --no-index -r requirements/computecanada_wheel.txt
```
Finally, it is necessary to make the project's package visible to the python interpreter. If the environment is to be
used by a single user, than the best solution is to simply install the package in editable mode with:
```bash
pip install -e .
```
However, if the virtual environment is shared between multiple users (e.g. installed somewhere in
`~/projects/def-pmjodoin` in order to be accessible to a whole team of people), then installing the package could
result in conflicts and users overwriting each other's installations. People might not even end up running the version
of the code they would expect!

In that case, the recommended approach is to indicate to the python interpreter *where* to look for the package right
before actually running scripts, by setting the PYTHONPATH environment variable. This would look something like this:
```bash
export PYTHONPATH=$PYTHONPATH:<PROJECT_ROOT_DIR>
python <your_script.py>
```
