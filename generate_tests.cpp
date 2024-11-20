#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <omp.h>

using namespace std;

const int MAX_SIZE = 20; // till 2^20
const int MAX_COEFFICIENT = 1e5;

void generate_random_polynomials(int n, vector<int> &poly1, vector<int> &poly2)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(0, MAX_COEFFICIENT);

    poly1.resize(n);
    poly2.resize(n);
    for (int j = 0; j < n; j++)
    {
        poly1[j] = dis(gen);
        poly2[j] = dis(gen);
    }
    if(poly1[0] == 0) poly1[0]++;
    if(poly2[0] == 0) poly2[0]++; 
}

vector<long long> multiply_polynomials(const vector<int> &poly1, const vector<int> &poly2)
{
    vector<long long> result(poly1.size() + poly2.size() - 1, 0);
    for (int i = 0; i < poly1.size(); i++)
    {
        for (int j = 0; j < poly2.size(); j++)
        {
            result[i + j] += static_cast<long long>(poly1[i]) * 1LL * poly2[j];
        }
    }
    return result;
}

int main()
{
    #pragma omp parallel for
    for (int pow = 0; pow <= MAX_SIZE; pow++)
    {
        int n = (1 << pow); 
        vector<int> poly1, poly2;
        generate_random_polynomials(n, poly1, poly2);
        vector<long long> result = multiply_polynomials(poly1, poly2);

        int test_case = pow + 1;
        ofstream file("tests/test_case_" + to_string(test_case) + ".txt");
        ofstream result_file("results/result_" + to_string(test_case) + ".txt");
        if (file)
        {
            file << n << "\n"; 
            for (int x : poly1)
                file << x << " ";
            file << "\n";
            for (int x : poly2)
                file << x << " ";

            for (long long x : result)
                result_file << x << " ";

            file.close();
        }
    }

    return 0;
}
