(1) Reduce_uint8.c : test code for MPI_SUM operation with type uint8.
    compile as : $path/mpicc -march=armv8-a+sve -O3 -o Reduce_uint8 Reduce_uint8.c 
(2) sve_uint8_test.sh: A shell script that will generate results with time information for MPI_SUM operation with different message size. 
(3) SVE_MPI_Op.py: A python script which you can use to generate plot.

ALL YOU NEED TO DO IS:
(1) Change the path of your mpirun and test binary in sve_uint8_test.sh. 
(2) Run sve_uint8_test.sh 
 This will generate 3 types of output files with names
    a. sve-sum-vectorlength.txt    ##MPI_SUM with SVE-enabled operation
    b. no-sve-sum-vectorlength.txt ##MPI_SUM without SVE-enabled operation
    c. sve-cpy-vectorlength.txt    ## memcpy 
(2) run python scripts as
    $python SVE_MPI_Op.py sve-sum-vectorlength.txt no-sve-sum-vectorlength.txt
    make sure put sve-sum-vectorlength.txt before no-sve-sum-vectorlength.txt 
    example : SVE_MPI_Op.py sve-sum-128.txt no-sve-sum-128.txt

