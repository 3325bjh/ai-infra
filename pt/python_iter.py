import sys
def countdown(n):
    while n>0:
        yield n
        n-=1
generator=countdown(5)
for value in generator:
    print(value)
print(next(generator))

class MyNumber:
    def __iter__(self):
        self.a=1
        return self
    def __next__(self):
        if self.a<=20:
            x=self.a
            self.a+=1
            return x
        else:
            raise StopIteration

myclass=MyNumber()
myiter=iter(myclass)
for x in myiter:
    print(x)
print(next(myiter))

list=[1,2,3,4]
it=iter(list)
print(next(it))
for x in it:
    print(x,end=" ")
while True:
    try:
        print(next(it))
    except StopIteration:
        sys.exit()

