# to run from within nocturne/cpp/
cd build
make
make install
cd ..
cd tests/build
make
cd ../bin
./test_simulation
cd ../..