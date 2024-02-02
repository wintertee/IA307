NVCC = nvcc

NVCCFLAGS = -Wno-deprecated-gpu-targets -lcublas

@echo "Usage: make <path>/<filename> (without extension)"

%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@.out

S3: S3/sgemm.cu  S3/fmatrix.cu  S3/cuda_stuff.cu S3/main.cu
	$(NVCC) $(NVCCFLAGS) $^ -o $@.out

clean:
	find . -type f -name '*.out' -delete
