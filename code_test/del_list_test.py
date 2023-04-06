list1 = [0, 35166, 20946]
new_group = []
for i in range(len(list1)):
    new_group.append(sum(list1[:i+1]))
print(new_group)
del new_group[0]
del new_group[-1]
print(new_group)
a = sum(list1[:2])
print(a)