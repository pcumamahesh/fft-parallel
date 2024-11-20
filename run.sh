#!/bin/bash
g++ -std=c++20 "polymul.cpp" -fopenmp -o "binaries/polymul"
g++ -std=c++20 "iterative_polymul.cpp" -fopenmp -o "binaries/iterative_polymul"
g++ -std=c++20 "polymul_parallel.cpp" -fopenmp -o "binaries/polymul_parallel"
g++ -std=c++20 "iterative_polymul_parallel.cpp" -fopenmp -o "binaries/iterative_polymul_parallel"

./binaries/polymul >results.txt
./binaries/iterative_polymul >>results.txt
./binaries/polymul_parallel >>results.txt
./binaries/iterative_polymul_parallel >>results.txt

python3 plot.py