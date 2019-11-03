class Student:
    def __init__(self,name,age,sex):
        self.name = name
        self.age = age
        self.sex = sex

    def show(self):
        print('name:',self.name,'  age:',self.age,'   sex:',self.sex)
