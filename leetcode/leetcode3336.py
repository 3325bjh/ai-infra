#https://leetcode.cn/problems/find-the-number-of-subsequences-with-equal-gcd/description/?envType=daily-question&envId=2026-07-14
from typing import List
from math import gcd
from collections import defaultdict


class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10 ** 9 + 7
        dp={(0,0):1}
        for num in nums:
            next_dp=defaultdict(int)
            for(g1,g2),count in dp.items():
                next_dp[g1,g2]=(
                    next_dp[g1,g2]+count
                )%MOD
                new_g1=gcd(g1,num)
                next_dp[new_g1,g2]=(
                    next_dp[new_g1,g2]+count
                )%MOD
                new_g2=gcd(g2,num)
                next_dp[g1,new_g2]=(
                    next_dp[g1,new_g2]+count
                )%MOD
            dp=next_dp
        return sum(
            count for(g1,g2),count in dp.items()
            if(g1==g2 and g1>0)
        )%MOD


