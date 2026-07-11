#https://leetcode.cn/problems/count-the-number-of-complete-components/?envType=daily-question&envId=2026-07-11
from typing import List
class Solution:
    def countCompleteComponents(self, n: int, edges: List[List[int]]) -> int:
        ans=0
        g=[[] for _ in range(n)]
        for a,b in edges:
            g[a].append(b)
            g[b].append(a)
        visit=[False for _ in range(n)]
        def dfs(x):
            visit[x]=True
            node_count = 1
            degree_sum = len(g[x])
            for y in g[x]:
                if not visit[y]:
                    sub_count, sub_degree = dfs(y)
                    node_count += sub_count
                    degree_sum += sub_degree
            return node_count, degree_sum
        for x in range(n):
            if not visit[x]:
                node_count, degree_sum = dfs(x)
                if degree_sum == node_count * (node_count - 1):
                    ans += 1
        return ans


