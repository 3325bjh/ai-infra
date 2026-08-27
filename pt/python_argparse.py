import argparse
class UpperAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.upper())
class CustomParser(argparse.ArgumentParser):
    def print_help(self):
        print("这是自定义帮助信息")
        super().print_help()
parser = argparse.ArgumentParser()
#参数类型
parser.add_argument('num', type=int, help='输入一个整数')
parser.add_argument('input_file', help='输入文件的路径')
#添加可选参数
parser.add_argument('-o', '--output', help='输出文件的路径')
#带默认值参数
parser.add_argument('-n', '--name', default='John', help='输入姓名，默认为 John')
parser.add_argument('--name2', action=UpperAction, help='输入姓名并转换为大写')
#参数组
group=parser.add_argument_group('文件操作参数')
group.add_argument('input_file', help='输入文件的路径')
group.add_argument('-o', '--output', help='输出文件的路径')
#互斥参数
group2 = parser.add_mutually_exclusive_group()
group2.add_argument('-v', '--verbose', action='store_true', help='详细输出')
group2.add_argument('-q', '--quiet', action='store_true', help='安静模式')
subparsers = parser.add_subparsers(title='子命令', dest='subcommand')

# 创建 create 子命令
create_parser = subparsers.add_parser('create', help='创建文件')
create_parser.add_argument('filename', help='要创建的文件名')

# 创建 delete 子命令
delete_parser = subparsers.add_parser('delete', help='删除文件')
delete_parser.add_argument('filename', help='要删除的文件名')
args = parser.parse_args()
#重写 ArgumentParser 的 print_help() 方法来自定义帮助信息的格式
parser2=CustomParser()
parser2.print_help()



