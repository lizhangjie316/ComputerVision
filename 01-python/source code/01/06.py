a_tuple = (1,5,32,17,63,2)
b_tuple = 2,4,5,7,1

a_list = [1,2,5,7,6,38,49,15,21]

for i in a_tuple:
    print(i)

for i in b_tuple:
    print(i)

for i in a_list:
    print(i)

for index in range(len(a_tuple)):
    print('index:',index,'number in tuple=',a_tuple[index])
