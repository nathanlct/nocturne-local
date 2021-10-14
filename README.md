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

Once done, if using Conda, first activate the environment where the Python library should be installed (eg. `conda activate nocturne`), then run the following to build and install the library:

```bash
# path to Python's libraries folder (site-packages)
# eg. /path/to/anaconda3/envs/nocturne/lib/python3.8/site-packages
DPYTHON_LIBRARY_DIR=$(python -c "import site; print(site.getsitepackages()[0])")
# path to Python executable
# eg. /path/to/anaconda3/envs/nocturne/bin/python3
DPYTHON_EXECUTABLE=$(which python)

mkdir build  # create build folder
cd build
# create Makefiles and download pybind11
cmake .. -DPYTHON_LIBRARY_DIR="${DPYTHON_LIBRARY_DIR}" \
    -DPYTHON_EXECUTABLE="${DPYTHON_EXECUTABLE}"
make  # build library
make install  # install library in Python's path
```

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
