# Example programs
cuda_add_executable(example main.cu
    OPTIONS ${GENCODE} ${VERBOSE_PTXAS})
