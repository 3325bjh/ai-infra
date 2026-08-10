from typing import List
class Solution:
    def validSequence(self, s: str, t: str) -> List[int]:
        n,m=len(s),len(t)
        suf=[0]*(n+1)
        suf[n]=m
        j=m-1
        for i in range(n-1,-1,-1):
            if j>=0 and s[i]==t[j]:
                j-=1
            suf[i]=j+1
        ans=[]
        changed=False
        j=0
        for i,c in enumerate(s):
            if c==t[j] or not changed and suf[i+1]<=j+1:
                if c!=t[j]:
                    changed=True
                ans.append(i)
                j+=1
                if j==m:
                    return ans
        return []