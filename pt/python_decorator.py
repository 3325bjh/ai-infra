def decorator_function(original_function):
    def wrapper(*args,**kwargs):
        print("执行前")
        result=original_function(*args,**kwargs)
        print("执行后")
        return result
    return wrapper
@decorator_function
def target_function():
    print("原函数执行")

#类装饰器
class SingletonDecorator:
    def __init__(self, cls):
        self.cls = cls
        self.instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = self.cls(*args, **kwargs)
        return self.instance

@SingletonDecorator
class Database:
    def __init__(self):
        print("初始化")

db1 = Database()
db2 = Database()
print(db1 is db2)

#内置装饰器
'''
@staticmethod：定义静态方法
@classmethod：定义类方法
@property：将方法变为属性
'''
class MyClass:
    @staticmethod
    def static_method():
        print("静态方法")

    @classmethod
    def class_method(cls):
        print(cls.__name__)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value