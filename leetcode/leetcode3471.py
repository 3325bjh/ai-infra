#https://leetcode.cn/problems/find-the-largest-almost-missing-integer/description/?envType=daily-question&envId=2026-08-18
from typing import List


class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        n=len(nums)
        if k==n:
            return max(nums)
        elif k==1:
            set1=set([])
            set2=set([])
            for num in nums:
                if num in set2:
                    continue
                if num in set1:
                    set1.remove(num)
                    set2.add(num)
                else:
                    set1.add(num)
            if len(set1)>0:
                return max(set1)
            else:
                return -1
        else:
            num1=nums[0]
            flag1=True
            num2=nums[n-1]
            flag2=True
            if num1==num2:
                return -1
            for i in range(1,n-1):
                if nums[i]==num1:
                    flag1=False
                if nums[i]==num2:
                    flag2=False
            if not flag1 and not flag2:
                return -1
            if (num1>num2 and flag1) or (flag1 and not flag2):
                return num1
            return num2