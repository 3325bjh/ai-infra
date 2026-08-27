'''
with expression [as variable]:
    # 代码块
'''

with open('input.txt', 'r') as infile, open('output.txt', 'w') as outfile:
    content = infile.read()
    outfile.write(content.upper())

#通过实现 __enter__ 和 __exit__ 方法创建自定义的上下文管理器：
class Timer:
    def __enter__(self):
        import time
        self.start=time.time()
        return self
    '''
    _exit__() 方法接收三个参数：
    exc_type：异常类型
    exc_val：异常值
    exc_tb：异常追踪信息
    '''
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end=time.time()
        print(f"耗时:{self.end-self.start:.2f}")

with Timer() as t:
    sum(range(100000))