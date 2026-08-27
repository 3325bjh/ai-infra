'''
lambda arguments: expression
- lambda是 Python 的关键字，用于定义 lambda 函数。
- arguments 是参数列表，可以包含零个或多个参数，但必须在冒号(:)前指定。
- expression 是一个表达式，用于计算并返回函数的结果。
'''

f=lambda : "Hello,world!"
print(f())
x=lambda a:a+10
print(x(5))

y=lambda a,b,c:a+b+c
print(y(5,6,2))

numbers = [1, 2, 3, 4, 5, 6, 7, 8]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # 输出：[2, 4, 6, 8]


