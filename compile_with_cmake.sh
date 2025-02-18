# clean and create build folder
rm -rf build && mkdir build && cd build

# run CMake
crun cmake ..

# build both executables
make ..

cd ..

# run tests via CTest
crun ctest --output-on-failure

./cuda_app

