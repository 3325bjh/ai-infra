#https://leetcode.cn/problems/predict-the-winner/description/?envType=daily-question&envId=2026-08-01
from collections import defaultdict
from typing import List
class Solution:
    def predictTheWinner(self, nums: List[int]) -> bool:
        n = len(nums)
        dp = {}

        def maxScore(i, j):
            if i > j:
                return 0
            if (i, j) in dp:
                return dp[(i, j)]

            # 当前玩家选左边，然后对手在剩余区间也最优选择
            left = nums[i] + min(maxScore(i + 2, j), maxScore(i + 1, j - 1))
            # 当前玩家选右边
            right = nums[j] + min(maxScore(i + 1, j - 1), maxScore(i, j - 2))

            dp[(i, j)] = max(left, right)
            return dp[(i, j)]

        score1 = maxScore(0, n - 1)
        total = sum(nums)
        return score1 >= total - score1