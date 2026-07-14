#https://leetcode.cn/problems/sequential-digits/description/?envType=daily-question&envId=2026-07-13
from typing import List
class Solution:
    def sequentialDigits(self, low: int, high: int) -> List[int]:
       ans=list()
       for i in range(1,10):
            num=i
            for j in range(i+1,10):
                num=num*10+j
                if(low<=num<=high):
                    ans.append(num)
       ans.sort()
       return ans

if __name__ == '__main__':
    for n in range(1,10):
        for start in range(1,10):
            num=0
            tmp=start
            for i in range(1,n+1):
                num=num*10+tmp
                if i==n:
                    print(num)
                tmp += 1
                if tmp == 10:
                    break


