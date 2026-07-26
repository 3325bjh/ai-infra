#https://leetcode.cn/problems/maximum-product-of-three-numbers/?envType=daily-question&envId=2026-07-26
from typing import List
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        nums.sort()
        n=len(nums)
        ans=nums[n-1]*nums[n-2]*nums[n-3]
        ans=max(nums[0]*nums[1]*nums[n-1],ans)
        return ans