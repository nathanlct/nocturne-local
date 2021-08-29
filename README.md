Nocturne is a 2D driving simulator, built in C++ for speed and exported as a Python library.

## Installation

Run the following to download the repo, build the Python library and install it. The paths in the `cmake` command should be replaced by the path where the Python library file should be installed, and the path of the Python executable (in this case, a conda environment named "nocturne" is used).

``` 
git clone https://github.com/nathanlct/nocturne.git
cd nocturne
mkdir build
cd build
cmake .. -DPYTHON_LIBRARY_DIR="/path/to/anaconda3/envs/nocturne/lib/python3.8/site-packages/" -DPYTHON_EXECUTABLE="/path/to/miniconda3/envs/nocturne/bin/python3"
make
make install
```

You should then be all set to use that library from your Python executable:

```
python
>>> from nocturne import Simulation
>>> sim = Simulation()
>>> sim.reset()
Resetting simulation.
```

Python tests can also be ran from `nocturne/tests`.

## C++ build instructions

To build the C++ library independently of the Python one, run the following:

```
cd nocturne/cpp
mkdir build
cd build
cmake ..
make
make install
```

Subsequently, the tests can be built using:

```
cd nocturne/cpp/tests
mkdir build
cd build
cmake ..
make
```

after which the executables can be found in `nocturne/cpp/tests/bin`.
