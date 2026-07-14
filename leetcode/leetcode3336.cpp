#include<vector>
#include<algorithm>
#include<numeric>
#include<unordered_map>
using namespace std;

class Solution {
public:
    int subsequencePairCount(vector<int>& nums) {
        const int MOD = 1'000'000'007;
        int n=nums.size();
        int m = *max_element(nums.begin(), nums.end());
        vector<vector<int>> dp(m+1,vector<int>(m+1));
        dp[0][0]=1;
        for(int num:nums){
            vector<vector<int>> ndp(m+1,vector<int>(m+1));
            for(int j=0;j<=m;j++){
                int divisor1 = gcd(j, num);
                for(int k=0;k<=m;k++){
                    int divisor2 = gcd(k, num);
                    ndp[j][k]=(ndp[j][k]+dp[j][k])%MOD;
                    ndp[divisor1][k] = (ndp[divisor1][k] + dp[j][k]) % MOD;
                    ndp[j][divisor2] = (ndp[j][divisor2] + dp[j][k]) % MOD;
                }
            }
            dp=ndp;
        }
        int ans=0;
        for(int i=0;i<=m;i++){
            ans=(ans+dp[i][i])%MOD;
        }
        return ans;
    }
};