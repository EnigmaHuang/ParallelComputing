#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
using namespace std;

int main(int argc, char** argv){
    int T = atoi(argv[1]);
    srand(0);
    for (int i = 1; i < T; ++i){
        int n = rand() % 60 + 1;
        int m = rand() % 60 + 1;
        double x1 = ((double)rand() / RAND_MAX - 0.5) * 4;
        double x2 = ((double)rand() / RAND_MAX - 0.5) * 4;
        if (x1 > x2) swap(x1, x2);
        double y1 = ((double)rand() / RAND_MAX - 0.5) * 4;
        double y2 = ((double)rand() / RAND_MAX - 0.5) * 4;
        if (y1 > y2) swap(y1, y2);
        char* command = new char[1111];
        printf("%d %.5f %.5f %.5f %.5f %d %d ans\n", 12, x1, x2, y1, y2, m, n);
        cout << "calc seq" << endl;
        sprintf(command, "./MS_seq %d %.5f %.5f %.5f %.5f %d %d ans", 12, x1, x2, y1, y2, m, n);
        system(command);
        cout << "end seq" << endl;
        cout << "calc parallel" << endl;
        sprintf(command, "mpirun -np 4 ./MS_hybrid %d %.5f %.5f %.5f %.5f %d %d out", 12, x1, x2, y1, y2, m, n);
        system(command);
        cout << "end parallel" << endl;
        system("diff out ans");
        cout << "Test #" << i << endl;
        bool res = system("diff out ans");
        if (res == false)
            cout << "No difference founded" << endl;
        else
            break;
        system("rm out");
    }
}