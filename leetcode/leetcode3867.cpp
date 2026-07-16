#include<vector>
#include<numeric>
#include<algorithm>
using namespace std;
class Solution {
public:
    long long gcdSum(vector<int>& nums) {
        vector<long> prefixGcd;
        int maxi=0;
        for(int &num:nums){
            maxi=max(maxi,num);
            prefixGcd.push_back(gcd(maxi,num));
        }
        sort(prefixGcd.begin(),prefixGcd.end());
        int i=0;
        long long sum=0;
        int j=prefixGcd.size()-1;
        while (i<j)
        {
            sum+=(gcd(prefixGcd[i],prefixGcd[j]));
            i++;
            j--;
        }
        return sum;
    }
};