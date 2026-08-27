class Solution:
    def checkDivisibility(self, n: int) -> bool:
        str_n=str(n)
        num_sum=0
        num_cheng=1
        for e in str_n:
            num_sum+=int(e)
            num_cheng*=int(e)
        return n%(num_sum+num_cheng)==0
