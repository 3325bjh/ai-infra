#https://leetcode.cn/problems/smallest-stable-index-i/?envType=daily-question&envId=2026-09-04
class Solution:
    def firstStableIndex(self, nums: list[int], k: int) -> int:
        pre_max=[]
        after_min=[]
        for num in nums:
            if len(pre_max)==0:
                pre_max.append(num)
            else:
                if num>pre_max[len(pre_max)-1]:
                    pre_max.append(num)
                else:
                    pre_max.append(len(pre_max)-1)
        for i in range(len(nums)-1,-1,-1):
            if len(after_min)==0:
                after_min.append(nums[i])
            else:
                if nums[i]<after_min[len(after_min)-1]:
                    after_min.append(nums[i])
                else:
                    after_min.append(len(after_min)-1)

        after_min.reverse()

        for i in range(len(nums)):
            if pre_max[i]-after_min[i]<=k:
                return i

        return -1


