#include<string>
#include<stack>
#include<unordered_map>
#include<unordered_set>
using namespace std;
class Solution {
public:
    string smallestSubsequence(string s) {
        unordered_map<char,int> map;
        int n=s.size();
        for(int i=0;i<n;i++){
            if(map.find(s[i])!=map.end()){
                map[s[i]]++;
            }else{
                map[s[i]]=1;
            }
        }
        string ans="";
        unordered_set<char> se;
        for(int i=0;i<n;i++){
            if(se.find(s[i])==se.end()){
                while (!ans.empty() && s[i]<ans.back() && map[ans.back()]>0) 
                {
                    se.erase(ans.back());
                    ans.pop_back();
                }
                ans.push_back(s[i]);
                se.insert(s[i]);
            }
            map[s[i]]--;
        }
        return ans;
    }
};