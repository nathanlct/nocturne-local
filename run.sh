DPYTHON_LIBRARY_DIR=$(python -c "import site; print(site.getsitepackages()[0])")
DPYTHON_EXECUTABLE=$(which python)

mkdir -p build
cd build
cmake .. -DPYTHON_LIBRARY_DIR="${DPYTHON_LIBRARY_DIR}" \
    -DPYTHON_EXECUTABLE="${DPYTHON_EXECUTABLE}"
make
make install
cd ..
