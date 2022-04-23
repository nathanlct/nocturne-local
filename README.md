Nocturne is a 2D driving simulator, built in C++ for speed and exported as a Python library.

## Installation

### Dependencies

Install CMake: 

Mac: `brew install cmake`

Linux: `sudo apt-get -y install cmake`

Nocturne also depends on SFML for drawing graphic elements and on pybind11 for building the Python library from C++ code. Both these libraries are automatically installed by CMake.

### Nocturne installation

```bash
# download repository
git clone https://github.com/nathanlct/nocturne.git
cd nocturne

# create a conda environment
conda env create -f environment.yml

# build and install nocturne into that environment
python setup.py develop
```

## Usage

```python
> python
Python 3.8.11
>>> from nocturne import Simulation
>>> sim = Simulation()
>>> sim.reset()
Resetting simulation.
```

## Common errors:

### CMAKE can't find SFML library.
Make sure the path to SFML is included in CMAKE_PREFIX_PATH.
### ImportError: libsfml-graphics.so.2.5: cannot open shared object file: No such file or directory
Make sure SFML/lib is included in LD_LIBRARY_PATH if you're on a linux machine 
### ImportError: libudev.so.0: cannot open shared object file
Do this really dumb thing. Make a folder, run ```ln -s /usr/lib/x86_64-linux-gnu/libudev.so.1 libudev.so.0``` then add that folder to the LD_LIBRARY_PATH
