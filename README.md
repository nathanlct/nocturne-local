Nocturne is a 2D driving simulator, built in C++ for speed and exported as a Python library.

It is currently designed to handle traffic scenarios from the [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset), and with some work could be extended to support different driving tasks. Using the Python library `nocturne`, one is able to train controllers for AVs to solve various tasks from the Waymo dataset, which we provide as a benchmark, then use the tools we offer to evaluate the designed controllers.

**Citation**: [Todo] 

# Installation

## Dependencies

[CMake](https://cmake.org/) is required to compile the C++ library. 

Run `cmake --version` to see whether CMake is already installed in your environment. If not, refer to the CMake website instructions for installation, or you can use:

- `sudo apt-get -y install cmake` (Linux)
- `brew install cmake` (MacOS)

Nocturne uses [SFML](https://github.com/SFML/SFML) for drawing and visualization, as well as on [pybind11](https://pybind11.readthedocs.io/en/latest/) for compiling the C++ code as a Python library.

To install SFML:

- `sudo apt-get install libsfml-dev` (Linux)
- `brew install sfml` (MacOS)

pybind11 is included as a submodule and will be installed in the next step.

## Nocturne

Start by cloning the repo:

```bash
git clone https://github.com/nathanlct/nocturne.git
cd nocturne
```

Then run the following to install git submodules:

```bash
git submodule sync
git submodule update --init
```

If you are using [Conda](https://docs.conda.io/en/latest/) (recommended), you can instantiate an environment and install Nocturne into it with the following:

```bash
# create the environment and install the dependencies
conda env create -f environment.yml

# activate the environment where the Python library should be installed
conda activate nocturne

# run the C++ build and install Nocturne into the simulation environment
python setup.py develop
```

If you are not using Conda, simply run the last command to build and install Nocturne at your default Python path.

You should then be all set to use that library from your Python executable:

```python
> python
Python 3.8.11
>>> from nocturne import Simulation
>>> sim = Simulation()
>>> sim.reset()
Resetting simulation.
```

Python tests can be ran with `pytest`.

## Constructing the dataset

At the moment for FAIR researchers the dataset is available on the cluster so no need to do anything.

## C++ build instructions

If you want to build the C++ library independently of the Python one, run the following:

```bash
cd nocturne/cpp
mkdir build
cd build
cmake ..
make
make install
```

Subsequently, the C++ tests can be ran with `./tests/nocturne_test` from within the `nocturne/cpp/build` directory.

## Common installation errors

### pybind11 installation errors

If you are getting errors with pybind11, install it directly in your conda environment (eg. `conda install -c conda-forge pybind11` or `pip install pybind11`, cf. https://pybind11.readthedocs.io/en/latest/installing.html for more info).

### CMake can't find SFML library.

Make sure the path to SFML is included in CMAKE_PREFIX_PATH.

### ImportError: libsfml-graphics.so.2.5: cannot open shared object file: No such file or directory

Make sure SFML/lib is included in LD_LIBRARY_PATH if you're on a linux machine 

### ImportError: libudev.so.0: cannot open shared object file

Do this really dumb thing. Make a folder, run ```ln -s /usr/lib/x86_64-linux-gnu/libudev.so.1 libudev.so.0``` then add that folder to the LD_LIBRARY_PATH
