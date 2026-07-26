#include<vector>
#include<algorithm>
using namespace std;
class Solution {
public:
    vector<int> gcdValues(vector<int>& nums, vector<long long>& queries) {
        int m=*max_element(nums.begin(),nums.end());
        vector<int> cnt_num(m+1);
        for(int &num:nums){
            cnt_num[num]++;
        }
        vector<long long> cnt_gcd(m+1);
        for(int i=m;i>0;i--){
            int c=0;
            for(int j=i;j<=m;j+=i){
                c+=cnt_num[j];
                cnt_gcd[i]-=cnt_gcd[j];
            }
            cnt_gcd[i]+= (long long)c * (c - 1) / 2; 
        }
        vector<long long> pre_sum(m+1);
        pre_sum[0]=0;
        for(int i=1;i<=m;i++){
            pre_sum[i]=cnt_gcd[i]+pre_sum[i-1];
        }
        vector<int> ans;
        for(long long &q:queries){
            ans.push_back(upper_bound(pre_sum.begin(),pre_sum.end(),q)-pre_sum.begin());
        }
        return ans;
    }
};