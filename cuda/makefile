generate : generate_data.cu
		nvcc -std=c++20 generate_data.cu -o generate_data rate_calculator.cu -ccbin=g++ -lm -O3
