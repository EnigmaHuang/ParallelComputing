/*
mpicxx -O3 -march=native -std=gnu++03 -lm -o HW2 HW2.cpp -fopenmp
mpirun -np 1 ./HW2 4 -2 2 -2 2 400 400 output_file
./MS_seq 4 -2 2 -2 2 400 400 out

mpirun -np 4 ./HW2 12 -2 2 -2 2 400 400 output_file

mpirun -np 4 ./HW2 1 -2 2 -2 2 4000 4000 output_file

mpirun -np 4 ./HW2 12 -2 2 -2 2 4000 4000 output_file
./MS_seq 4 -2 2 -2 2 4000 4000 out4000

mpirun -np 4 ./HW2 12 -1.5 1.5 -1.5 1.5 800 800 output_file
./MS_seq 4 -2 2 -2 2 3 3 out
mpirun -np 1 ./HW2 4 -2 2 -2 2 3 3 output_file

./MS_draw output_file
*/