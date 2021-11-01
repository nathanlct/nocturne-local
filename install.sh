# path to Python's libraries folder (site-packages)
# eg. /path/to/anaconda3/envs/nocturne/lib/python3.8/site-packages
DPYTHON_LIBRARY_DIR=$(python -c "import site; print(site.getsitepackages()[0])")
# path to Python executable
# eg. /path/to/anaconda3/envs/nocturne/bin/python3
DPYTHON_EXECUTABLE=$(which python)
pybind11_DIR=$(pwd)

mkdir -p build  # create build folder
cd build
# create Makefiles and download pybind11
cmake .. -DPYTHON_LIBRARY_DIR="${DPYTHON_LIBRARY_DIR}" \
    -DPYTHON_EXECUTABLE="${DPYTHON_EXECUTABLE}"
make  # build library
make install  # install library in Python's path
