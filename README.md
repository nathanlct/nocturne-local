Nocturne is a 2D driving simulator, built in C++ for speed and exported as a Python library.

## Installation

Start by downloading the repo:

```bash
git clone https://github.com/nathanlct/nocturne.git
cd nocturne
```

Nocturne uses [SFML](https://github.com/SFML/SFML) for visualization, it can be installed with:

-   Linux: `sudo apt-get install libsfml-dev`
-   MacOS: `brew install sfml`

If you are using Conda to build the environments, you can instantiate the nocturne environment by running `conda env create -f environment.yml`.
Once done, if using Conda, first activate the environment where the Python library should be installed (eg. `conda activate nocturne`), then run the following to build and install the library: `python setup.py develop`. This will run the C++ build and install Nocturne
into your simulation environment.

Note: if you are getting errors with pybind11, install it directly in your conda environment (eg. `conda install -c conda-forge pybind11` or `pip install pybind11`, cf. https://pybind11.readthedocs.io/en/latest/installing.html for more info).

You should then be all set to use that library from your Python executable:

```python
> python
Python 3.8.11
>>> from nocturne import Simulation
>>> sim = Simulation()
>>> sim.reset()
Resetting simulation.
```

Python tests can also be ran from `nocturne/tests`.

## Constructing the dataset
At the moment for FAIR researchers the dataset is available on the cluster so no need to do anything.

## C++ build instructions

To build the C++ library independently of the Python one, run the following:

```bash
cd nocturne/cpp
mkdir build
cd build
cmake ..
make
make install
```

Subsequently, the tests can be built using:

```bash
cd nocturne/cpp/tests
mkdir build
cd build
cmake ..
make
```

after which the executables can be found in `nocturne/cpp/tests/bin`.

## Common errors:
### CMAKE can't find SFML library.
Make sure the path to SFML is included in CMAKE_PREFIX_PATH.
### ImportError: libsfml-graphics.so.2.5: cannot open shared object file: No such file or directory
Make sure SFML/lib is included in LD_LIBRARY_PATH if you're on a linux machine 
### ImportError: libudev.so.0: cannot open shared object file
Do this really dumb thing. Make a folder, run ```ln -s /usr/lib/x86_64-linux-gnu/libudev.so.1 libudev.so.0``` then add that folder to the LD_LIBRARY_PATH
