#https://leetcode.cn/problems/remove-covered-intervals/?envType=daily-question&envId=2026-07-06
from typing import  List
class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x:(x[0],-x[1]))
        n=len(intervals)
        ans=n
        right=intervals[0][1]
        for i in range(1,n):
            if intervals[i][1]<=right:
                ans=ans-1
            else:
                right=intervals[i][1]
        return ans
