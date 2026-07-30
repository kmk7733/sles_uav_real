## catkin_python_setup() reads this to install the `mpc_controller` python package
## (the simulation modules under src/mpc_controller/). Do NOT run it directly.
from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['mpc_controller'],
    package_dir={'': 'src'},
)

setup(**setup_args)
