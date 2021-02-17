import numpy as np

l = [1 ,[]]
l[1].append([2, []])
l[1].append([3, []])
l[1].append([4, []])
l[1][0][1].append([5, []])

print(l)
print(l[0])
print(l[1])
print(l[1][0])
print(l[1][0][0])
print(l[1][0][1])
print(l[1][1])
print(l[1][1][0])
print(l[1][1][1])
print(l[1][2])
print(l[1][2][0])
print(l[1][2][1])
