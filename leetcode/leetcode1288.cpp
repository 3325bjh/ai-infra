using namespace std;
#include<vector>
#include<algorithm>
class Solution {
public:
    int removeCoveredIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(),intervals.end(),[](vector<int>&a,vector<int>&b){
            if(a[0]!=b[0]){
                return a[0]<b[0];
            }
            return a[1]>b[1];
        });
        int n=intervals.size();
        int ans=n;
        int right=intervals[0][1];
        for (int i=1;i<n;i++){
            if (intervals[i][1]<=right){
                ans--;
            }else{
                right=intervals[i][1];
            }
        }
        return ans;
    }
};