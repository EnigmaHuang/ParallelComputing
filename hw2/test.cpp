#include <iostream>
using namespace std;

int main(){
    int a[3] = {1, 2, 3};
    reverse(a, a + 2);
    for (int i = 0; i < 3; ++i)
        printf("%d\n", a[i]);
}