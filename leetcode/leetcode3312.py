#https://leetcode.cn/problems/sorted-gcd-pair-queries/?envType=daily-question&envId=2026-07-17
from bisect import bisect_right
from itertools import accumulate
from typing import List
class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        max_m=max(nums)
        cnt_x=[0]*(max_m+1)
        cnt_gcd=[0]*(max_m+1)
        for num in nums:
            cnt_x[num]+=1
        for i in range(max_m,0,-1):
            cnt_c=0
            for j in range(i,max_m+1,i):
                cnt_c+=cnt_x[j]
                cnt_gcd[i]-=cnt_gcd[j]
            cnt_gcd[i]+=(cnt_c*(cnt_c-1))/2
        s= list(accumulate(cnt_gcd))
        return [bisect_right(s,q) for q in queries]

