#https://leetcode.cn/problems/rank-transform-of-an-array/description/?envType=daily-question&envId=2026-07-12
from typing import List
class Solution:
    def arrayRankTransform(self, arr: List[int]) -> List[int]:
        rank= {num:i+1 for i,num in enumerate(sorted(set(arr)))}
        return [rank[num] for num in arr]