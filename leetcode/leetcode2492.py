# https://leetcode.cn/problems/minimum-score-of-a-path-between-two-cities/description/?envType=daily-question&envId=2026-07-04
import heapq
import math
from queue import PriorityQueue
from typing import List

class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        """
        :type n: int
        :type roads: List[List[int]]
        :rtype: int
        """
        self.p=[i for i in range(n+1)]
        self.m=[100000 for _ in range(n+1)]
        for a,b,distance in roads:
            pa=self.findP(a)
            pb=self.findP(b)
            if(pa==pb):
                self.m[pa]=min(self.m[pa],distance)
            else:
                self.p[pa]=pb
                self.m[pb]=min(self.m[pa],min(distance,self.m[pb]))
        return self.m[self.findP(1)]

    def findP(self,x):
        if self.p[x]!=x:
            self.p[x]=self.findP(self.p[x])
        return self.p[x]


class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        """
        :type n: int
        :type roads: List[Lis
        """
        visit=[False for _ in range(n+1)]
        graph=[[] for _ in range(n+1)]
        for a,b,distance in roads:
            graph[a].append((b,distance))
            graph[b].append((a,distance))

        ans=math.inf

        def dfs(x):
            nonlocal ans
            visit[x]=True
            for v,d in graph[x]:
                ans=min(ans,d)
                if visit[v] is False:
                    dfs(v)
        dfs(1)
        return ans


class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        """
        :type n: int
        :type roads: List[Lis
        """
        graph=[[] for _ in range (n+1)]
        visit=[False for _ in range(n+1)]
        # q=PriorityQueue()
        q = []
        for a,b,distance in roads:
            graph[a].append((distance,b))
            graph[b].append((distance,a))
            if a==1:
                heapq.heappush(q,(distance,b))
                # q.put((distance,b))
            if b==1:
                heapq.heappush(q,(distance,a))
                # q.put((distance,a))
        ans=math.inf
        visit[1]=True
        while q:
            # d,p=q.get()
            d,p=heapq.heappop(q)
            if visit[p]:
                continue
            visit[p]=True
            ans=min(ans,d)
            for distance,to in graph[p]:
                # q.put((distance,to))
                heapq.heappush(q,(distance,to))

        return ans

