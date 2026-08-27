from collections import defaultdict
from typing import List
class Solution:
    def maxNumberOfFamilies(self, n: int, reservedSeats: List[List[int]]) -> int:
        seats=defaultdict(int)
        for row,seat in reservedSeats:
            if 2<=seat<=9:
                seats[row] |=1<<(seat-2)
        empty_rows=n-len(seats)
        ans=empty_rows*2
        for x in seats.values():
            if x&0b1111==0 or x&0b111100==0 or x&0b11110000==0:
                ans+=1
        return ans