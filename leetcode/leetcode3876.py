#https://leetcode.cn/problems/construct-uniform-parity-array-ii/?envType=daily-question&envId=2026-09-03
class Solution:
    def uniformArray(self, nums1: list[int]) -> bool:
        INF = 10 ** 9  
        min_odd = INF
        min_even = INF
        for e in nums1:
            if e%2==0:
                min_even=min(min_even,e)
            else:
                min_odd=min(min_odd,e)

        return min_odd==1000000 or min_even==1000000 or min_even>min_odd
