#include<vector>
#include<set>
#include<unordered_map>
#include<algorithm>
using namespace std;
class Solution {
public:
    vector<int> arrayRankTransform(vector<int>& arr) {
        vector<int> arrSorted=arr;
        vector<int> ans=vector<int>(arr.size());
        sort(arrSorted.begin(),arrSorted.end());
        unordered_map<int,int> rank;
        for(auto a:arrSorted){
            if(!rank.count(a)){
                rank[a]=rank.size()+1;
            }
        }
        for(int i=0;i<arr.size();i++){
            ans[i]=rank[arr[i]];
        }
        return ans;
    }
};