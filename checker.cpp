#pragma once

#include <iostream> 
#include <fstream>
#include <string> 
#include <vector> 
#include <climits> 

using namespace std; 

long long get_error(vector<long long> &result, int test_case)
{
    long long error = 0;
    vector<long long> correct_result;

    ifstream correct_file("results/result_" + to_string(test_case) + ".txt");

    int n = result.size(); 
    for(int i = 0; i < n; ++i)
    {
        long long x; 
        correct_file >> x; 
        error = max(error, abs(x - result[i]));
    }
    return error; 
}