from typing import List
class Solution:
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        n=len(stoneValue)
        sumScore=sum(stoneValue)
        def maxStone(i)->int:
            if i>=n:
                return 0
            maxScore=stoneValue[i]+min(maxStone(i+2),min(maxStone(i+3),maxStone(i+4)))
            if i<n-1:
                maxScore=max(maxScore,stoneValue[i]+stoneValue[i+1]+min(maxStone(i+3),min(maxStone(i+4),maxStone(i+5))))
            if i<n-2:
                maxScore=max(maxScore,stoneValue[i]+stoneValue[i+1]+stoneValue[i+2]+min(maxStone(i+4),min(maxStone(i+5),maxStone(i+6))))
            return maxScore
        AliceScore=maxStone(0)
        BobScore=sumScore-AliceScore
        if AliceScore>BobScore:
            return "Alice"
        elif AliceScore==BobScore:
            return "Tie"
        else:
            return "Bob"

class Solution:
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        n=len(stoneValue)
        dp=[0]*(n+4)
        sumScore=sum(stoneValue)
        for i in range(n-1,-1,-1):
            dp[i]=stoneValue[i]+min(dp[i+2],dp[i+3],dp[i+4])
            if i<n-1:
                dp[i]=max(dp[i],stoneValue[i]+stoneValue[i+1]+min(dp[i+3],dp[i+4],dp[i+5]))
            if i<n-2:
                dp[i]=max(dp[i],stoneValue[i]+stoneValue[i+1]+stoneValue[i+2]+min(dp[i+4],dp[i+5],dp[i+6]))
        AliceScore=dp[0]
        BobScore=sumScore-AliceScore
        if AliceScore>BobScore:
            return "Alice"
        elif AliceScore==BobScore:
            return "Tie"
        else:
            return "Bob"

if __name__ == '__main__':
    s=Solution()
    s.stoneGameIII([1,2,3,7])