try:
    file = open('eeee','r+')
except Exception as e:
    print('There is no file named as eeee')
    response = input("do you want to create a eeee file?(y/n) ")
    if response == 'y':
        file = open('eeee','w')
    else:
        pass
else:
    file.write('hahahaha')

file.close()
    


