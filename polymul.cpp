#include <iostream>
#include <vector>
#include <chrono>
#include <complex>
#include <cmath>

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

void fft(vector<Complex> &a)
{
    int n = a.size();
    if (n == 1)
        return;

    int half = n / 2;

    vector<Complex> a_odd(half), a_even(half);

    for (int i = 0; i < half; ++i)
    {
        a_even[i] = a[2 * i];
        a_odd[i] = a[2 * i + 1];
    }

    fft(a_even);
    fft(a_odd);

    long double angle = 2 * PI / n;

    for (int i = 0; i < half; ++i)
    {
        Complex w_i = Complex(cosl(angle * i), sinl(angle * i));
        a[i] = a_even[i] + w_i * a_odd[i];
        a[i + n / 2] = a_even[i] - w_i * a_odd[i];
    }
}

void inverse_fft(vector<Complex> &a)
{
    int n = a.size();
    if (n == 1)
        return;

    int half = n / 2;

    vector<Complex> a_odd(half), a_even(half);

    for (int i = 0; i < half; ++i)
    {
        a_even[i] = a[2 * i];
        a_odd[i] = a[2 * i + 1];
    }

    inverse_fft(a_even);
    inverse_fft(a_odd);

    long double angle = -2 * PI / n;

    for (int i = 0; i < half; ++i)
    {
        Complex w_i = Complex(cosl(angle * i), sinl(angle * i));
        a[i] = a_even[i] + w_i * a_odd[i];
        a[i + n / 2] = a_even[i] - w_i * a_odd[i];
        a[i] /= 2;
        a[i + n / 2] /= 2;
    }
}

pair<vector<i64>, i64> polymul(vector<int> &poly1, vector<int> &poly2)
{
    int n = poly1.size();
    int sz = 1;
    while (sz < poly1.size() + poly2.size())
        sz *= 2;

    vector<Complex> points1(sz), points2(sz);
    vector<i64> result(2 * n - 1);

    for (int i = 0; i < n; i++)
    {
        points1[i] = Complex(poly1[i], 0);
        points2[i] = Complex(poly2[i], 0);
    }

    auto start = getTime();
    fft(points1);
    fft(points2);

    for (int i = 0; i < sz; ++i)
    {
        points1[i] *= points2[i];
    }

    inverse_fft(points1);

    for (int i = 0; i < 2 * n - 1; i++)
    {
        result[i] = roundl(points1[i].real());
    }
    auto end = getTime();
    auto duration = getDuration(start, end);
    return {result, duration};
}

int main()
{
    int test_cases = 19;
    cout << "Recursive Polymul: ";
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