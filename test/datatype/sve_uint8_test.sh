echo "ompi version with SVE -- Usage: arg1: count of elements, args2: 'i'|'f'|'d' : datatype: integer, float, double. args3 size of type. args4 operation"
echo "your_path/mpirun -mca op sve -mca op_sve_hardware_available 1  -np 1 /your_test_path/Reduce_local_float 1048576  i 8 max"

# test all vector size
for vector_len in  128 256 512 1024 2048
do

    echo "=========Integer type all operations & all sizes========"
    echo ""
    echo ""
    echo -e "Test \e[1;33m SVE full vector instruction for loop \e[m Total_num_bits = 2048*N "
    for (( i=1; i<31; i++ ))
    do
        for val in 1024 4096 16384 65536 262144 1048576 4194304 16777216 33554432
        do
            /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/build/bin/mpirun -mca op sve -mca op_sve_hardware_available 1 -mca pml ob1  -np 1  armie -msve-vector-bits=$vector_len --iclient libinscount_emulated.so  --unsafe-ldstex   --  /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/test/datatype/Reduce_uint8  $val i 8 sum |  tee -a sve-sum-$vector_len.txt
            /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/build/bin/mpirun -mca op sve -mca op_sve_hardware_available 0 -mca pml ob1  -np 1  armie -msve-vector-bits=$vector_len --iclient libinscount_emulated.so  --unsafe-ldstex   --  /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/test/datatype/Reduce_uint8  $val i 8 sum |   tee -a no-sve-sum-$vector_len.txt
            /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/build/bin/mpirun -mca op sve -mca op_sve_hardware_available 1 -mca pml ob1  -np 1  armie -msve-vector-bits=$vector_len --iclient libinscount_emulated.so  --unsafe-ldstex   --  /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/test/datatype/Reduce_uint8  $val i 8 cpy |  tee -a sve-cpy-$vector_len.txt 
        done
    done
done
