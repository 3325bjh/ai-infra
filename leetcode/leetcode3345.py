#https://leetcode.cn/problems/smallest-divisible-digit-product-i/?envType=daily-question&envId=2026-08-06
class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        def check(x):
            c=1
            while x!=0:
               mod=x%10
               c*=mod
               x=x//10
            if c%t==0:
                return True
            return False
        for i in range(0,10):
            if check(n):
                return n
            n+=1