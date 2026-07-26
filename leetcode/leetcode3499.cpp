#include<string>
#include<vector>
using namespace std;
class Solution {
public:
    int maxActiveSectionsAfterTrade(string s) {
        vector<int> zero;
        int i=0;
        int ans=0;
        while (i<s.size())
        {
            if(s[i]=='0'){
                int num=0;
                while (s[i]=='0')
                {
                    num++;
                    i++;
                }
                zero.push_back(num);
            }else{
                i++;
                ans++;
            }
        }
        int maxNum=0;
        if (zero.empty()){
            return ans;
        }
        for(i=0;i<zero.size()-1;i++){
            maxNum=max(maxNum,zero[i]+zero[i+1]);
        }
        return ans+maxNum;
    }
};