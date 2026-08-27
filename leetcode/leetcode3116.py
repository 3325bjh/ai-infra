#https://leetcode.cn/problems/kth-smallest-amount-with-single-denomination-combination/?envType=daily-question&envId=2026-08-21
import math
from typing import List

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        n = len(coins)

        # 预先计算每个非空子集的 lcm 以及容斥符号
        subsets = []

        for mask in range(1, 1 << n):
            lcm_value = 1
            bits = 0

            for i in range(n):
                if mask & (1 << i):
                    bits += 1
                    lcm_value = lcm_value // math.gcd(
                        lcm_value, coins[i]
                    ) * coins[i]

            # 奇数个硬币：加上
            # 偶数个硬币：减去
            sign = 1 if bits % 2 == 1 else -1
            subsets.append((lcm_value, sign))

        def count(x: int) -> int:
            """统计 <= x 的不同金额数量"""
            rank = 0

            for lcm_value, sign in subsets:
                if lcm_value > x:
                    continue

                rank += sign * (x // lcm_value)

            return rank

        left = 1
        right = min(coins) * k

        # 查找第一个满足 count(x) >= k 的 x
        while left < right:
            mid = left + (right - left) // 2

            if count(mid) >= k:
                right = mid
            else:
                left = mid + 1

        return left