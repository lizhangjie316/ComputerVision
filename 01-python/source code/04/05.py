import pandas as pd

#定义资料集并打印出
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                             'A': ['A0', 'A1', 'A2', 'A3'],
                             'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                              'C': ['C0', 'C1', 'C2', 'C3'],
                              'D': ['D0', 'D1', 'D2', 'D3']})

print(left)
#    A   B key
# 0  A0  B0  K0
# 1  A1  B1  K1
# 2  A2  B2  K2
# 3  A3  B3  K3

print(right)
#    C   D key
# 0  C0  D0  K0
# 1  C1  D1  K1
# 2  C2  D2  K2
# 3  C3  D3  K3

#依据key column合并，并打印出
res = pd.merge(left, right, on='key')

print(res)
#    A   B   key C   D
# 0  A0  B0  K0  C0  D0
# 1  A1  B1  K1  C1  D1
# 2  A2  B2  K2  C2  D2
# 3  A3  B3  K3  C3  D3
