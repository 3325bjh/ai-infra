from typing import List
class SegmentTree:
    def SegmentTree(self,arr:List):
        self.n=len(arr)
        self.arr=arr
        self.seg=[0]*(self.n<<2)
        self.build(1,0,self.n-1)


    def build(self,p,l,r):
        if l==r:
            self.seg[p]=self.arr[l]
            return
        mid=(l+r)>>1
        self.build(p*2,l,mid)
        self.build(p*2+1,mid+1,r)
        self.seg[p]=max(
            self.seg[p*2],self.seg[p*2+1]
        )
    def query(self,L,R):
        def _query(p,l,r):
            if L<=l and r<=R:
                return self.seg[p]
            mid=(l+r)>>1
            res=0
            if L<=mid:
                res=max(res,_query(2*p,l,mid))
            if R>mid:
                res=max(res,_query(2*p+1,mid+1,r))
            return res

        return _query(1,0,self.n-1)