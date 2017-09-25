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
        int n = rand() % 10 + 1;
        int C = rand() % 10 + 1;
        int T = rand() % 20 + 1;
        int N = rand() % 100 + 1;
        cout << "Test #" << i;
        printf(": %d %d %d %d\n", n, C, T, N);
        char command[1111];
        sprintf(command, "./force %d %d %d %d ans", n, C, T, N);
        system(command);
        sprintf(command, "./roller %d %d %d %d out", n, C, T, N);
        system(command);
        system("diff out ans");
        bool res = system("diff out ans");
        if (res == false)
            cout << "No difference founded" << endl;
        else
            break;
        system("rm out");
    }
}