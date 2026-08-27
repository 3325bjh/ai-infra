#https://leetcode.cn/problems/distribute-elements-into-two-arrays-i/?envType=daily-question&envId=2026-08-20
from typing import List
class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        array1=[nums[0]]
        array2=[nums[1]]
        n=len(nums)
        for i in range(2,n):
            if array1[len(array1)-1]>array2[len(array2)-1]:
                array1.append(nums[i])
            else:
                array2.append(nums[i])
        return array1+array2