#https://leetcode.cn/problems/concatenate-non-zero-digits-and-multiply-by-sum-i/description/?envType=daily-question&envId=2026-07-07
class Solution:
    def sumAndMultiply(self, n: int) -> int:
        x=""
        sum=0
        while n!=0:
            mod=n%10
            if mod != 0:
                sum+=mod
                x=str(mod)+x
            n=n//10
        return 0 if x == "" else sum * int(x)

class Solution:
    def sumAndMultiply(self, n: int) -> int:
        x=str(n).replace("0","")
        if x == "":
            return 0

        digit_sum = sum(int(ch) for ch in x)
        return digit_sum * int(x)

