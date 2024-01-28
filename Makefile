NVCC = nvcc

NVCCFLAGS = -I./include -Wno-deprecated-gpu-targets

@echo "Usage: make <path>/<filename> (without extension)"

%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@.out

clean:
	find . -type f -name '*.out' -delete
