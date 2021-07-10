
# Compilers
NVCC=$(PREP) nvcc

# Flags

CUDACFLAGS  = -I${CUDA_PATH}/include

CUDALDFLAGS   = -L${CUDA_PATH}/lib64 -lcudart

SW2D-GPU: SW2D-GPU.cu Makefile
	$(NVCC) $(CUDACFLAGS) $(CUDALDFLAGS) SW2D-GPU.cu -o SW2D-GPU

print:
	@echo ""
	@echo "Compling..."

done:
	@echo ""
	@echo "Done!!!!"
