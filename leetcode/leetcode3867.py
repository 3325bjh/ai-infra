#https://leetcode.cn/problems/sum-of-gcd-of-formed-pairs/?envType=daily-question&envId=2026-07-16
import math
class Solution:
    def gcdSum(self, nums: list[int]) -> int:
        prefixGcd=[]
        maxi=0
        for num in nums:
            maxi=max(maxi,num)
            prefixGcd.append(math.gcd(maxi,num))
        prefixGcd.sort()
        i=0
        sum=0
        j=len(prefixGcd)-1
        while i<j:
            sum+=math.gcd(prefixGcd[i],prefixGcd[j])
            i+=1
            j-=1
        return sum
