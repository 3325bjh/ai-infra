import json
'''
json.dumps(): 对数据进行编码。
json.loads(): 对数据进行解码。
'''
data1={
    'no':1,
    'name':'Runoob',
    'url':'http://www.runoob.com'
}
json_str = json.dumps(data1)
print("Python 原始数据：", repr(data1))
print("JSON 对象：", json_str)

# 将 JSON 对象转换为 Python 字典
data2 = json.loads(json_str)
print("data2['name']: ", data2['name'])
print("data2['url']: ", data2['url'])