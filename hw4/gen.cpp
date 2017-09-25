#include <iostream>
#include <cstdio>
#include <cstdlib>
using namespace std;

int main(int argc, char** argv){
    freopen("in", "w", stdout);
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int T = atoi(argv[3]);
    srand(T);
    printf("%d %d\n", n, m);
    for (int i = 0; i < m; ++i){
        printf("%d %d %d\n", rand() % n + 1, rand() % n + 1, rand() % 101);
    }
}