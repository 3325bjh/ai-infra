#https://leetcode.cn/problems/find-greatest-common-divisor-of-array/?envType=daily-question&envId=2026-07-18
import math
from typing import List
class Solution:
    def findGCD(self, nums: List[int]) -> int:
        num_min=min(nums)
        num_max=max(nums)
        return math.gcd(num_min,num_max)
