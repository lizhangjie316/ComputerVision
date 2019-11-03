import copy

a = 10
b = copy.deepcopy(a)


print(id(a)==id(b))


