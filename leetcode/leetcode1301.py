from typing import List
#https://leetcode.cn/problems/number-of-paths-with-max-score/?envType=daily-question&envId=2026-07-05
class Solution:
    def pathsWithMaxScore(self, board: List[str]) -> List[int]:
        n= len(board)
        dp=[[[-1,0] for _ in range(n) ] for _ in range (n)]
        dp[n-1][n-1][0]=0
        def update(x,y,u,v):
            if u>=n or v>=n or dp[u][v][0]==-1:
                return
            if dp[u][v][0]>dp[x][y][0]:
                dp[x][y]=dp[u][v][:]
            elif dp[u][v][0]==dp[x][y][0]:
                dp[x][y][1]+=dp[u][v][1]
        for i in range(n-1,-1,-1):
            for j in range(n-1,-1,-1):
                if board[i][j]!="S" and board[i][j]!="X":
                    update(i,j,i+1,j)
                    update(i,j,i,j+1)
                    update(i,j,i+1,j+1)
                    if dp[i][j][0]!=-1 and board[i][j]!="E":
                        dp[i][j][0]+=int(board[i][j])

        return [dp[0][0][0], dp[0][0][1] % (10 ** 9 + 7)] if dp[0][0][0] != -1 else [0, 0]




