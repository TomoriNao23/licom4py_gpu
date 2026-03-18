# earthlab
# module purge

# # c/fortran.intel
# module load compiler/intel/2017.5.239

# # mpi.intel
# # module load mpi/intelmpi/2017.4.239
# # mpi.openmpi
# module load mpi/openmpi/intel/4.1.6
# # mpi.hpcx
# # module load mpi/hpcx/2.7.4/intel-2017.5.239

# # netcdf.c/fortran.intel
# module load mathlib/netcdf/intel/4.4.1

# #cmake
# module load compiler/cmake/3.20.1 

# # cmake -DCMAKE_C_COMPILER=icc -DCMAKE_Fortran_COMPILER=ifort  ..
# module load conda/base
# conda activate jax_chtholly


# tianjing
module purge
module load sghpc-mpi-gcc/25.8
use_conda
conda activate jax_dcu
#conda activate jax_dcu
export JAX_ENABLE_X64=1
export JAX_TRACEBACK_FILTERING=off
#python -c "import jax; print(jax.devices())"

#cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran ..