class Solution {
public:
    int gcdOfOddEvenSums(int n) {
        int sumOdd=n*n;
        int sumEven=(n+1)*n;
        return gcd(sumEven,sumOdd);
    }
};