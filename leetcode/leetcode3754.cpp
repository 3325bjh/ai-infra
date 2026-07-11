using namespace std;
#include<string>
class Solution {
public:
    long long sumAndMultiply(int n) {
        long long x = 0;
        int digitSum = 0;

        string s = to_string(n);

        for (char c : s) {
            if (c != '0') {
                int digit = c - '0';
                digitSum += digit;
                x = x * 10 + digit;
            }
        }

        return x * digitSum;
    }
};