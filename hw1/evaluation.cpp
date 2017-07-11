#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
using namespace std;

int main(int argc, char** argv){
    int T = atoi(argv[1]);
    int n, m, seed;
    for (int i = 1; i < T; ++i){
        seed = rand() % 100;
        n = rand();
        char command[111];
        memset(command, 0, sizeof command);
        if (seed <= 40) n = n % 10;
        if (seed <= 99 && seed >= 41) n = n % 100000;
        if (seed == 100) n = n % 100000000;
        m = rand() % 24 + 1;
        sprintf(command, "./gen %d", n);
        system(command);
        sprintf(command, "./force %d", n);
        system(command);
        sprintf(command, "mpirun -np %d ./HW1_x1054028 %d input output", m, n);
        system(command);
        cout << "Test #" << i << endl;
        bool res = system("diff output ans");
        if (res == false)
            cout << "No difference founded" << endl;
        else
            break;
        system("rm output");
    }
}