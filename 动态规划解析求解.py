'''
动态规划策略评估解析求解
Yiheng
'''
import numpy as np

'''
state:s1娱乐，s2上课，s3上课，s4写论文，s5睡觉
action:玩，学习，写作，发表，睡觉
'''
num_st = 5
num_action = 5

policy = np.zeros((num_st, num_action))
policy[0][:2] = [0.5,0.5]  # policy1是娱乐，policy1涉及到的action有玩和学习
policy[1][:2] = [0.5,0.5]  # policy2是上课，policy2涉及到的action有玩和学习
policy[2][2:4] = [0.5,0.5]  # policy3是上课，policy3涉及到的action有睡觉和写作
policy[3][3:] = [0.5,0.5]  # policy4是写论文，policy4涉及到的action有玩睡觉和发表
# policy5是最终状态不涉及任何转变
# print(policy)  # 打印策略矩阵，策略本来就是一个选择action的过程

'''
构建动态特性函数
'''
P = dict()  # 动态特性函数

for state in range(num_st - 1):
    for action in range(num_action):
        P[(state,action)] = [(0,0,0)]

# 这里state和action的索引都是从0开始的（state, reward, probability）
P[(0, 0)] = [(0, -1, 1)]
P[(0, 1)] = [(1, 0, 1)]
P[(1, 0)] = [(0, -1, 1)]
P[(1, 1)] = [(2, -2, 1)]
P[(2, 2)] = [(3, -2, 1)]
P[(2, 4)] = [(4, 0, 1)]
P[(3, 4)] = [(4, 10, 1)]
P[(3, 3)] = [(1, 1, 0.2)]
P[(3, 3)].append((2, 1, 0.4))
P[(3, 3)].append((3, 1, 0.4))


'''
构建方程组AX=b，X表示价值函数向量
'''
b = np.zeros((num_action))
gamma = 1
# 求取所有状态对应的贝尔曼方程里的常数项
for state in range(num_st - 1):
    for action in range(num_action):
        pi = policy[state][action]
        for p_item in P[(state, action)]:
            b[state] += pi * p_item[1] * p_item[2]

#  求A，也就是将方程组用矩阵形式表示
A = np.eye(num_st)  # 单位矩阵，行代表的是当前的是当前状态，列代表的是下一时刻的转化状态
gamma = 1

for state in range(num_st - 1):
    for action in range(num_action):
        pi = policy[state][action]
        for p_item in P[(state, action)]:
            next_st, reward, prob = p_item
            A[state][next_st] -= gamma * pi * prob  # 这里之所以是减，是因为将非常数部分转移到了等式左边

v = np.linalg.solve(A, b)  # numpy求齐次方程组的解，求出来的直接是各个状态对应的v

'''
求解q
'''
q = np.zeros((num_st, num_action))
for state in range(num_st - 1):
    for action in range(num_action):
        for p_item in P[(state,action)]:
            next_st, reward, prob = p_item
            q[state][action] += prob * (reward + gamma * v[next_st])  # 这里是求q用的
