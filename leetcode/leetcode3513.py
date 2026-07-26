#https://leetcode.cn/problems/number-of-unique-xor-triplets-i/description/?envType=daily-question&envId=2026-07-23
from typing import List
class Solution:
    def uniqueXorTriplets(self, nums: List[int]) -> int:
        n=len(nums)
        if n<=2:
            return n
        ans=1
        while ans<=n:
            ans<<=1
        return ans