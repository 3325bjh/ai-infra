#https://leetcode.cn/problems/minimum-number-of-pushes-to-type-word-ii/?envType=daily-question&envId=2026-07-31
class Solution:
    def minimumPushes(self, word: str) -> int:
        ans=0
        map=dict()
        for e in word:
            if map.get(e)==None:
                map[e]=1
            else:
                map[e]+=1
        map=dict(sorted(map.items(),key=lambda x:x[1],reverse=True))
        i=1
        for val in map.values():
            if i<=8:
                ans+=val
            if 8 < i <= 16:
                ans+=val*2
            elif 16<i<=24:
                ans+=val*3
            elif 24<i<=32:
                ans+=val*4
            i+=1
        return ans


if __name__ == '__main__':
    s=Solution()
    s.minimumPushes("aabbccddeeffgghhiiiiii")