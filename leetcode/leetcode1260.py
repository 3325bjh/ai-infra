#https://leetcode.cn/problems/shift-2d-grid/?envType=daily-question&envId=2026-07-20
from typing import List
class Solution:
    def shiftGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        m=len(grid)
        n=len(grid[0])
        ans=[[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                index=(i*n+j+k)%(m*n)
                i_put=index//n
                j_put=index%n
                ans[i_put][j_put]=grid[i][j]

        return ans