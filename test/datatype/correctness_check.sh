echo "ompi version with SVE -- Usage: arg1: count of elements, args2: 'i'|'f'|'d' : datatype: integer, float, double. args3 size of type. args4 operation"
echo "your_path/mpirun -mca op sve -mca op_sve_hardware_available 1  -np 1 /your_test_path/Reduce_local_float 1048576  i 8 max"

Orange= "\033[0;33m"
Blue= '\033[0;34m'
Purple= '\033[0;35m'

NC='\033[0m'

# test all vector size
for vector_len in  128 256 512 1024 2048 
do

    echo "=========Integer type all operations & all sizes========"
    echo ""
    for op in max min sum mul band bor bxor
    do
        echo ""
        echo "===Operation  $op test==="
        for size in 8 16 32 64
        do
            echo -e "Test \e[1;33m SVE full vector instruction for loop \e[m Total_num_bits = 2048*N "
            /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/build/bin/mpirun -mca op sve -mca op_sve_hardware_available 1 -mca pml ob1  -np 1  armie -msve-vector-bits=$vector_len --iclient libinscount_emulated.so  --unsafe-ldstex   --  /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/test/datatype/Reduce_local_float  1024 i $size $op 
            echo -e "Test \e[1;34m SVE partial vector instruction for loop$ \e[m (2048*(N-1) ) <  Total_num_bits < 2048*N "
            /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/build/bin/mpirun -mca op sve -mca op_sve_hardware_available 1 -mca pml ob1  -np 1  armie -msve-vector-bits=$vector_len --iclient libinscount_emulated.so  --unsafe-ldstex   --  /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/test/datatype/Reduce_local_float  1040 i $size $op 
        done
    done

    echo "=======Float type all operations========"
    echo ""
    echo -e "Test \e[1;33m SVE full vector instruction for loop \e[m Total_num_bits = 2048*N "
    for op in max min sum mul
    do
        /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/build/bin/mpirun -mca op sve -mca op_sve_hardware_available 1  -mca pml ob1  -np 1  armie -msve-vector-bits=$vector_len --iclient libinscount_emulated.so  --unsafe-ldstex   --  /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/test/datatype/Reduce_local_float 1024 f 32 $op 
    done

    echo -e "Test \e[1;34m SVE partial vector instruction for loop$ \e[m (2048*(N-1) ) <  Total_num_bits < 2048*N "
    for op in max min sum mul
    do
        /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/build/bin/mpirun -mca op sve -mca op_sve_hardware_available 1  -mca pml ob1  -np 1  armie -msve-vector-bits=$vector_len --iclient libinscount_emulated.so  --unsafe-ldstex   --  /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/test/datatype/Reduce_local_float 1040 f 32 $op
    done


    echo "========Double type all operations========="
    echo ""

    echo -e "Test \e[1;33m SVE full vector instruction for loop \e[m Total_num_bits = 2048*N "
    for op in max min sum mul
    do
        /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/build/bin/mpirun -mca op sve -mca op_sve_hardware_available 1 -mca pml ob1  -np 1  armie -msve-vector-bits=$vector_len --iclient libinscount_emulated.so  --unsafe-ldstex   --  /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/test/datatype/Reduce_local_float 1024 d 64 $op
    done

    echo -e "Test \e[1;34m SVE partial vector instruction for loop$ \e[m (2048*(N-1) ) <  Total_num_bits < 2048*N "
    for op in max min sum mul
    do
        /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/build/bin/mpirun -mca op sve -mca op_sve_hardware_available 1 -mca pml ob1  -np 1  armie -msve-vector-bits=$vector_len --iclient libinscount_emulated.so  --unsafe-ldstex   --  /ccsopen/home/dzhong/Downloads/github/intel_to_arm/ompi/test/datatype/Reduce_local_float 1040 d 64 $op
    done
done
##echo -e "Test \e[1;35m  duff device code \e[m  2048*N <  Total_num_bits < 2048*N + 256  "
