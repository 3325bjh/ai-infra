from typing import List
class Solution:
    def stoneGameII(self, piles: List[int]) -> int:
        n = len(piles)
        suffix_sum = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + piles[i]

        memo = {}

        def dfs(i: int, M: int) -> int:
            # 如果已经取完所有石子
            if i >= n:
                return 0

            if i + 2 * M >= n:
                return suffix_sum[i]

            if (i, M) in memo:
                return memo[(i, M)]

            max_stones = 0
            current_sum = 0

            for X in range(1, 2 * M + 1):
                if i + X > n:
                    break
                current_sum += piles[i + X - 1]
                opponent = dfs(i + X, max(M, X))
                max_stones = max(max_stones, current_sum + (suffix_sum[i + X] - opponent))

            memo[(i, M)] = max_stones
            return max_stones

        return dfs(0, 1)
