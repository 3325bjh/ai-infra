#https://leetcode.cn/problems/smallest-subsequence-of-distinct-characters/?envType=daily-question&envId=2026-07-19
class Solution:
    def smallestSubsequence(self, s: str) -> str:
        dict={}
        instack=set()
        for e in s:
            if e in dict:
                dict[e]+=1
            else:
                dict[e]=1
        stack=[]
        for e in s:
            if e not in instack:
                while stack and e<stack[len(stack)-1] and dict[stack[len(stack)-1]]>0:
                    instack.remove(stack.pop())
                stack.append(e)
                instack.add(e)
            dict[e]-=1

        return "".join(stack)
