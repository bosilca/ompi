mkdir build

./autogen.pl >/dev/null

./configure --prefix=$PWD/build --enable-mpirun-prefix-by-default --enable-debug CC=armclang CFLAGS="-march=armv8-a+sve" CXX=armclang++ FC=armflang >/dev/null

./config.status >/dev/null
make -j 128 install >/dev/null

## compile the test code, test code under ompi/test/datapyte/Reduce_local_float.c
./build/bin/mpicc  -g  -O3 -march=armv8-a+sve  -o ./test/datatype/Reduce_local_float ./test/datatype/Reduce_local_float.c
