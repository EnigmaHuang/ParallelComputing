#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cstdio>
#include <algorithm>
using namespace std;

int main(int argc, char** argv){
    int N = atoi(argv[1]);
    std::ifstream fin("input", std::ios::binary);
    float *chunk = new float[N];
    fin.read((char*)chunk, sizeof(float) * N);
    sort(chunk, chunk + N);
    std::ofstream fout("ans", std::ios::binary);
    fout.write((char*)chunk, sizeof(float) * N);
}