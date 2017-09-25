#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
using namespace std;

int main(int argc, char** argv){
    int T = atoi(argv[1]);
    char* exe = argv[2];
    srand(0);
    for (int i = 1; i < T; ++i){
        int tp = rand() % 3;
        int n;
        if (tp == 0) n = rand() % 20 + 1;
        if (tp == 1) n = rand() % 100 + 1;
        if (tp == 2) n = rand() % 1000 + 1;
        int m = rand() % (n * n) + n;
        cout << "Test #" << i;
        printf(": %d %d\n", n, m);
        char command[1111];
        sprintf(command, "./gen %d %d %d\n", n, m, T);
        system(command);
        sprintf(command, "cuda-memcheck ./%s in out 16", exe);
        system(command);
        sprintf(command, "./seq_FW.exe in ans");
        system(command);
        bool res = system("diff out ans");
        if (res == false)
            cout << "No difference founded" << endl;
        else
            break;
        system("rm out");
        system("rm ans");
    }
}
