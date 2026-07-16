#https://leetcode.cn/problems/gcd-of-odd-and-even-sums/submissions/735965086/?envType=daily-question&envId=2026-07-15
import math
class Solution:
    def gcdOfOddEvenSums(self, n: int) -> int:
        sumOdd=n*n
        sumEven=(n+1)*n
        return math.gcd(sumOdd,sumEven)
    def gcd(self,a,b):
        if b==0:
            return a
        return self.gcd(b,a%b)