#include <iostream>
#include "mpi.h"

using namespace std;

int main() {
    MPI_Init();
    
    MPI_Finalize();
}