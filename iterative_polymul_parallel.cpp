#include <iostream>
#include <vector>
#include <chrono>
#include <complex>
#include <cmath>
#include <omp.h>

#include "checker.cpp"

using namespace std;
using namespace std::chrono;

using Complex = complex<long double>;
using i64 = long long;
const long double PI = acosl(-1);

auto getTime()
{
    return high_resolution_clock::now();
}

auto getDuration(time_point<high_resolution_clock> start, time_point<high_resolution_clock> end)
{
    return duration_cast<microseconds>(end - start).count();
}

int bit_reverse(int x, int bitwidth)
{
    int reversed = 0;
    for (int i = 0; i < bitwidth; i++)
    {
        reversed |= ((x >> i) & 1) << (bitwidth - i - 1);
    }
    return reversed;
}

void fft(vector<Complex> &leaves)
{
    int n = leaves.size();
    for (int len = 2; len <= n; len <<= 1)
    {
        long double angle = 2 * PI / len;
        Complex wlen(cosl(angle), sinl(angle));
        #pragma omp parallel for
        for (int i = 0; i < n; i += len)
        {
            Complex w(1);
            #pragma omp parallel for
            for (int j = 0; j < len / 2; ++j)
            {
                Complex w(cosl(angle * j), sinl(angle * j));
                Complex u = leaves[i + j];
                Complex v = leaves[i + j + len / 2] * w;
                leaves[i + j] = u + v;
                leaves[i + j + len / 2] = u - v;
            }
        }
    }
}

void inverse_fft(vector<Complex> &leaves)
{
    int n = leaves.size();
    for (int len = 2; len <= n; len <<= 1)
    {
        long double angle = -2 * PI / len;
        Complex wlen(cosl(angle), sinl(angle));
        #pragma omp parallel for
        for (int i = 0; i < n; i += len)
        {
            Complex w(1);
            #pragma omp parallel for
            for (int j = 0; j < len / 2; ++j)
            {
                Complex w(cosl(angle * j), sinl(angle * j));
                Complex u = leaves[i + j];
                Complex v = leaves[i + j + len / 2] * w;
                leaves[i + j] = u + v;
                leaves[i + j + len / 2] = u - v;
            }
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        leaves[i] /= n;
}

pair<vector<i64>, i64> polymul(vector<int> &poly1, vector<int> &poly2)
{
    int n = poly1.size();
    int sz = 1;
    while (sz < poly1.size() + poly2.size())
        sz *= 2;

    vector<Complex> points1(sz), points2(sz);
    vector<Complex> bit_reversed_points1(sz), bit_reversed_points2(sz);
    vector<Complex> product(sz);
    vector<Complex> bit_reversed_product(sz);
    vector<i64> result(2 * n - 1);

    for (int i = 0; i < n; i++)
    {
        points1[i] = Complex(poly1[i], 0);
        points2[i] = Complex(poly2[i], 0);
    }

    int bitwidth = __lg(sz);

    auto start = getTime();

    #pragma omp parallel for
    for (int i = 0; i < sz; ++i)
    {
        bit_reversed_points1[i] = points1[bit_reverse(i, bitwidth)];
        bit_reversed_points2[i] = points2[bit_reverse(i, bitwidth)];
    }

    fft(bit_reversed_points1);
    fft(bit_reversed_points2);

    #pragma omp parallel for
    for (int i = 0; i < sz; ++i)
    {
        product[i] = bit_reversed_points1[i] * bit_reversed_points2[i];
    }

    #pragma omp parallel for
    for (int i = 0; i < sz; ++i)
    {
        bit_reversed_product[i] = product[bit_reverse(i, bitwidth)];
    }

    inverse_fft(bit_reversed_product);

    #pragma omp parallel for
    for (int i = 0; i < 2 * n - 1; i++)
    {
        result[i] = roundl(bit_reversed_product[i].real());
    }
    auto end = getTime();
    auto duration = getDuration(start, end);
    return {result, duration};
}

int main()
{
    int test_cases = 19;
    cout << "Iterative Polymul Parallel: ";
    cout << "\n";
    cout << "TEST \t ERROR \t TIME (microsec)\n";
    for (int test_case = 1; test_case <= test_cases; test_case++)
    {
        ifstream file("tests/test_case_" + to_string(test_case) + ".txt");
        int n;
        file >> n;
        vector<int> poly1(n), poly2(n);
        for (int i = 0; i < n; i++)
            file >> poly1[i];
        for (int i = 0; i < n; i++)
            file >> poly2[i];

        auto result = polymul(poly1, poly2);
        auto product = result.first;
        auto duration = result.second;

        i64 error = get_error(product, test_case);

        // cout << duration << " ";
        cout << test_case << " \t " << error << " \t " << duration << "\n";
    }
    cout << "\n";
}