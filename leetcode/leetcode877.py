#https://leetcode.cn/problems/stone-game/description/?envType=daily-question&envId=2026-08-02
from typing import List
class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        n=len(piles)
        dp=[[0]*n for _ in range(n)]
        sumStone=sum(piles)
        for idx,num in enumerate(piles):
            dp[idx][idx]=num
        for i in range(n-1,-1,-1):
            for j in range(i+1,n):
                dp[i][j]=max(piles[i]+min(dp[i+2][j] if i+2<n else 0,dp[i+1][j-1] if i+1<n and j-1>=0 else 0),
                             piles[j]+min(dp[i+1][j-1] if i+1<n and j-1>=0 else 0,dp[i][j-2] if j-2>=0 else 0))

        return dp[0][n-1]>=sumStone-dp[0][n-1]
