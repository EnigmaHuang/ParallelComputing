#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cstdio>
using namespace std;

int T;

int main(int argc, char** argv){
    int N = atoi(argv[1]);
    freopen("T", "r", stdin);
    std::cin >> T;
    fclose(stdin);
    srand(T++);
    freopen("T", "w", stdout);
    std::cout << T;
    fclose(stdout);
    float *chunk = new float[N];
    freopen("out", "w", stdout);
    for (int i = 0; i < N; ++i){
        chunk[i] = rand() / 65536.0;
        cout << chunk[i] << endl;
    }
    fclose(stdout);
    std::ofstream fout("input", 
    std::ios::binary);
    fout.write((char*)chunk, sizeof(float) * N);
}