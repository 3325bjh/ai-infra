names = ['Bob','Tom','alice','Jerry','Wendy','Smith']
new_names = [name.upper()for name in names if len(name)>3]
print(new_names)
#列表推导式
multiples = [i for i in range(30) if i % 3 == 0]
print(multiples)
listdemo = ['Google','Runoob', 'Taobao']
#字典推导式
newdict={key:len(key) for key in listdemo}
#集合推导式
setnew={i**2 for i in (1,2,3)}
#元组推导式
a=(x for x in range(1,10))