using namespace std;
#include<vector>
#include<string>
class Solution {
public:
    int mod=1000000000+7;
    vector<int> pathsWithMaxScore(vector<string>& board) {
        int n=board.size();
        vector<vector<vector<int>>> dp=vector<vector<vector<int>>>(n,vector<vector<int>>(n,vector<int>{-1,0}));
        dp[n-1][n-1][0]=0;
        dp[n-1][n-1][1]=1;
        for (int i=n-1;i>=0;i--){
            for(int j=n-1;j>=0;j--){
                if (board[i][j]!='X' && board[i][j]!='S'){
                    update(dp,i,j,i+1,j,n);
                    update(dp,i,j,i+1,j+1,n);
                    update(dp,i,j,i,j+1,n);
                    if (dp[i][j][0]!=-1 && board[i][j]!='E'){
                        dp[i][j][0]+=(board[i][j]-'0');
                    } 
                }
            }
        }
        return dp[0][0][0]==-1?vector<int>{0,0}:dp[0][0];
    }

    void update(vector<vector<vector<int>>> &dp,int x,int y,int u,int v,int n){
        if (u>=n || v>=n || dp[u][v][0]==-1){
            return;
        }
        if(dp[u][v][0]>dp[x][y][0]){
            dp[x][y][0]=dp[u][v][0];
            dp[x][y][1]=dp[u][v][1];
        }else if (dp[u][v][0]==dp[x][y][0])
        {
            dp[x][y][1]=(dp[x][y][1]+dp[u][v][1])%mod;
        }
    
    }
};