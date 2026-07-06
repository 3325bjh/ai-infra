// https://leetcode.cn/problems/minimum-score-of-a-path-between-two-cities/description/?envType=daily-question&envId=2026-07-04
#include <vector>
using namespace std;
class Solution
{
public:
    vector<int> p;
    vector<int> m;
    int minScore(int n, vector<vector<int>> &roads)
    {
        for (int i = 0; i <= n + 1; i++)
        {
            p.push_back(i);
        }
        for (int i = 0; i <= n + 1; i++)
        {
            m.push_back(100000);
        }
        for (int i = 0; i < roads.size(); i++)
        {
            int a = roads[i][0];
            int b = roads[i][1];
            int c = roads[i][2];
            int pa = findP(a);
            int pb = findP(b);
            if (pa == pb)
            {
                m[pa] = min(m[pa], c);
            }
            else
            {
                p[pa] = pb;
                m[pb] = min(m[pa], min(c, m[pb]));
            }
        }
        return m[findP(1)];
    }
    int findP(int x)
    {
        if (p[x] != x)
        {
            p[x] = findP(p[x]);
        }
        return p[x];
    }
};