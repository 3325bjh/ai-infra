#https://leetcode.cn/problems/maximize-active-section-with-trade-i/?envType=daily-question&envId=2026-07-21
class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        n = len(s)
        cnt1 = s.count('1')

        zeroBlocks = []
        i = 0
        while i < n:
            start = i

            while i < n and s[i] == s[start]:
                i += 1

            if s[start] == '0':
                zeroBlocks.append(i - start)

        m = len(zeroBlocks)

        if m < 2:
            return cnt1

        bestGain = 0  # 最优增量
        for i in range(m - 1):
            bestGain = max(bestGain, zeroBlocks[i] + zeroBlocks[i + 1])
        return cnt1 + bestGain

