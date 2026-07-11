#include<vector>
#include<utility>
using namespace std;
class Solution {
public:
    vector<vector<int>> g;
    vector<bool> visited;
    int countCompleteComponents(int n, vector<vector<int>>& edges) {
        g=vector<vector<int>>(n);
        visited=vector<bool>(n);
        int ans=0;
        for (int i = 0; i < edges.size(); i++)
        {
            int a=edges[i][0];
            int b=edges[i][1];
            g[a].push_back(b);
            g[b].push_back(a);
        }
        for (int i=0;i<n;i++){
            if(!visited[i]){
                pair<int,int> res=dfs(i);
                if (res.second==res.first*(res.first-1)){
                    ans++;
                }
            }
        }
        return ans;
    }
    pair<int,int> dfs(int x){
        visited[x]=true;
        pair<int,int> res;
        res.first=1;
        res.second=g[x].size();
        for (int i=0;i<g[x].size();i++){
            if(!visited[g[x][i]]){
                pair<int,int> count=dfs(g[x][i]);
                res.first+=count.first;
                res.second+=count.second;
            }
        }
        return res;
    }
};