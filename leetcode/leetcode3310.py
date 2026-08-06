#https://leetcode.cn/problems/remove-methods-from-project/?envType=daily-question&envId=2026-08-05
from typing import List
class Solution:
    def remainingMethods(self, n: int, k: int, invocations: List[List[int]]) -> List[int]:
        ans=[i for i in range(n)]
        g:List[List[int]]=[[] for _ in range(n) ]
        indegree=[0]*n
        for a,b in invocations:
            g[a].append(b)
            indegree[b]+=1
        needRemove=set()
        def dfs(x:int):
            needRemove.add(x)
            for e in g[x]:
                indegree[e]-=1
                if e not in needRemove:
                    dfs(e)
        dfs(k)
        for e in needRemove:
            if indegree[e]>0:
                    return ans
        return [e for e in ans if e not in needRemove]