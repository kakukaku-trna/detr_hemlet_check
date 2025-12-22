import random
# print("Hello World")
# print("你好");print("hi yjj")
# a = 11
# b = 2.666
# c = "yjj"
# a_float = float(a)
# b_str = str(b)
# b_str = b_str.replace("2.", "11")
# print(type(b_str))
# a_type = type(a)
# print(a, b, c, a+b, c )
# print(a_float, b_str)
# print(type(a), type(b), type(c))
# print(a_type, type("a"))
# money = 50
# money -= 10
# print("当前金额剩余",money)
# money -= 5
# print("当前金额剩余",money)
# print("1+1=", 1 + 1)
# print("2-4=", 2 - 4)
# print("3 ^ 3=", 3 ^ 3)
# print("99/13=", 99 / 13)
# print("99 // 13=", 99 // 13)
# print("99 % 13=", 99 % 13)
# print("13 * 7 + 8=", 13 * 7 + 8)
# name = '小杨\' '
# name_yang = ("y=x+1/x"
#              "77")
# name_and = name + name_yang
# name_li = ("""杨
#             洋""")
# print(name_yang, name, name_li, name_and)
# name_yang = name_yang.upper()
# print(name_yang)
#
# number1 = "111"
# number2 = "222"
# number3 = "111 %s%s "%(number1, number2)
# print(number1, number2, number3)
# print(f"1*1的结果 是{1*1}")
# print("1*1 的结果类型是%s"%(type(1*1)))


##杨某公司股票， 字符串格式化
# name = "杨某"
# stock_price = 19.99
# stock_code = "003032"
# print(f"公司：{name}，股票代码：{stock_code},当前股价:{stock_price}")
# stock_price_daily_growth_factor = float(input())1
# growth_days = float(input())
# a = stock_price*stock_price_daily_growth_factor**growth_days
# # print("每日增长系数：%s ，经过%s天的增长，股价达到了%.2fs"%(stock_price_daily_growth_factor,growth_days,a))
# #杨某游乐园，条件语句
# print("欢迎来到杨某游乐场 ")
# name = input("请输入您的姓名：")
# age = int(input("请输入您的年龄"))
# vip_name = 'xiaoyang'
#
# if name == vip_name:
#     print("祝小杨小朋友玩的愉快，您是超级无敌vip")
# elif age < 18:
#     print("小朋友玩的愉快")
# else:
#     print("您已成年，票价30")
# #杨某随机数小游戏
# import random
# num_random = random.randint(1, 99)
# #print(num_random)
# i = 1
# lunci = 20
# while i <= lunci:
#     a = int(input("请输入你所猜测的数字："))
#     if a == num_random:
#         print(f'Yes，恭喜你第{i}次就成功了')
#         break
#     if a > num_random:
#         print('请再试一次,这次有点大了')
#     else:
#         print('请再试一次,这次有点小了')
#     i += 1
# if i > lunci:
#    print('No,你失败了%s次，这次的数字是%s'%(lunci, num_random))

# #小杨表白
# i = 0
# a = 0
# while i < 100:
#     i += 1
#     a += i
#     print("小杨，我喜欢你 ，%s,%s" %(i,a))
#     #

# #杨某一直猜

# num_random = random.randint(1, 99)
# #print(num_random)
# i = 1
# while  i != 0:
#     i += 1
#     a = int(input("请输入你所猜测的数字："))
#     if a == num_random:
#         print(f'Yes，恭喜你第{i}次就成功了')
#         i = 0
#     else:
#       if a > num_random:
#         print('请再试一次,这次有点大了')
#       else:
#         print('请再试一次,这次有点小了')

# #小杨乘法表
# i = 1
# while i < 10 :
#     j = i
#     while j < 10 :
#         print(f"\t {i}*{j} = {i*j}  ", end='')
#         j += 1
#     print("  ")
#     i += 1

# for i in range(10):
#     for j in range(1,i+1):
#         print(f"{j}*{i} = {j*i} \t",end='')
#     print()

# #数一数有几个杨（ang）
# story = "从前有个杨某，是杨家村的特别喜欢杨树，每天都给村里的杨树浇水"
# x = "杨"
# a = 0
# for i in story:
#     if i == x:
#         a += 1
# print(a)

# #0-100有多少偶数
# i = 0
# for x in range(101):
#     a = x % 2
#     if a == 0 :
#         i = i + 1
#         print(x)
# print(i)

# #小杨发工资
# acc = 10000
# num_yuangong = 20
#
# for i in range(1,num_yuangong+1):
#     prp = random.randint(1,10)
#     if acc <= 0:
#         print("工资发完了，下个月领取吧")
#         break
#     if prp < 5:
#         print(f"员工{i}，绩效分为{prp}，低于5分，不发工资")
#         continue
#     else:
#         acc -= 1000
#         print(f"员工{i}发放工资1000元，账户余额剩余{acc}元")

# # #数字符串长度
# #
# def shushu(str):
#     a = 0
#     for i in str:
#       a = a + 1
#     print(a)
#     return (a)
#
# str = input()
# lenth = shushu(str)
# print(lenth)

# def add_m(x,y):
#     return x + y
# a = 1
# b = 2
# result = add_m(a,b)
# print(result)

# def xiaoyangtiwen(temperature):
#     if temperature <= 37.5:
#         print(f"欢迎来到小杨游乐园，请您出示核算和健康码 \n 您的体温是{temperature}，属于正常范围，\n祝您游玩愉快")
#     else:
#         print(f"欢迎来到小杨游乐园，请您出示核算和健康码 \n 您的体温是{temperature}，属于高热，\n需要隔离")
#     return
# temperature = float(input())
# xiaoyangtiwen(temperature)

# #小杨银行：
# def balance(canshu1):
#     if canshu1 == 1:
#         print("-------------------------查询余额-------------------------")
#     """
#     查询银行账户余额
#     name：户主姓名
#     """
#     x = "小杨"
#     if name == x:
#         print(f"欢迎小杨宝贝，您好，您的账户余额是{account}")
#     else:
#         print(f"欢迎{name}，您好，您的账户余额是{account}")
#     return
# def access_cun():
#     """
#     存钱
#     name：户主姓名
#     account 账户余额
#     money 存入金额
#     """
#     money = int(input("您要存入的金额："))
#     global account
#     account += money
#     print("-------------------------存款-------------------------")
#     print(f"{name},您好，您存款{money}元成功！")
#     balance(0)
#     return
# def access_qu():
#     """
#     取钱
#     name：户主姓名
#     account 账户余额
#     money 取出金额
#     """
#     print("-------------------------取款-------------------------")
#     money = int(input("您要取出的金额："))
#     global account
#     account -= money
#     print(f"{name},您好，您取款{money}元成功！")
#     balance(0)
#     return
#
# x = "小杨"
# print("尊贵的用户您好，欢迎来到小杨银行")
# name = input("尊贵的用户您好，请输入您的姓名：")
# x = "小杨"
# if name == x:
#     account = 50000000
# else:
#     account = 1000000
# num_bool = 1
# while num_bool > 0:
#     print("-------------------------主菜单-------------------------")
#     print(f"尊贵{name}您好，欢迎来到小杨银行\n查询余额【请输入1】\n存款【请输入2】\n查询余额【请输入1】\n取款【请输入3】\n退出【请输入4】")
#     a = int(input())
#     if a == 1:
#         balance(1)
#     if a == 2:
#         access_cun()
#     if a == 3:
#         access_qu()
#     if a == 4:
#         break

# #小杨列表
# age = [21,25,21,23,22,20]
# #2.追加数字31到列表尾部
# age.append(31)
# print(f"追加数字31到列表尾部,{age}")
# #3.追加一个新列表[29,33,30]
# age.extend([29,33,30])
# print(f"追加一个新列表[29,33,30],{age}")
# #4.取出第一个元素 21
# first_yua = age[0]
# print(f"取出第一个元素,{first_yua}")
# #5.取出最后一个元素 30
# last_yua = age[-1]
# print(f"取出最后一个元素,{last_yua}")
# #6.查找元素31 在列表中的下标位置
# weizhi = age.index(31)
# print(f"查找元素31 在列表中的下标位置,{weizhi}")
# for i in age:
#     print(i)

# num = [1,2,3,4,5,6,7,8,9,10]
# #while循环
# def list_while_xunhua (num_list):
#     num2 = []
#     i = 0
#     while i < len(num_list):
#         if num_list[i] % 2 == 0:
#             num2.append(num_list[i])
#             i += 1
#         else :
#             i += 1
#     return num2
# def list_for_xunhua (num_list):
#     num2 = []
#     for i in num_list:
#         if i % 2 == 0:
#             num2.append(i)
#     return num2
# num_list_1 = list_while_xunhua(num)
# num_list_2 = list_for_xunhua(num)
# print(f"while循环结果{num_list_1}\n for循环结果{num_list_2}")

# #元组
# t1 = ('小杨', 11, ['qinqin','xiaoyang'])
# #1.查询年龄的下标位置
# position = t1.index(11)
# print(f"年龄的下标位置{position}")
# #2.查询学生姓名
# name = t1.index('小杨')
# print(name)
# print(f"姓名{t1[name]}", )
# #3.删除xiaoyang
# t1[2].remove('xiaoyang')
# print(f"删除xiaoyang{t1}")
# #4.增加xiaoli
# t1[2].append('xiaoli')
# print(f"增加xiaoli{t1}")

#分割字符串
# #存在给定字符串
# name = "xiaoli love xiaoyang"
# #统计字符串内有多少xi字符
# a = name.count('xi')
# print(f"统计字符串内有多少xi字符:{a}")
# #将字符串中的空格全部替换为：“|”
# name2 = name.replace(' ','|')
# print(f"将字符串中的空格全部替换为：“|”:{name2}")
# #按照|对字符串进行分割
# name_list = name2.split("|")
# print(f"按照|对字符串进行分割:{name_list}")
# print(type(name_list))

# #序列切片
# list = [1,"xiaoli","love","xiaoyang",5,6,"xiaoyang"]
# str = "xiaoyang shi dabiantia"
# tuple = (1,2,3,4,5,6,"xiaoyang")
# #1.list列表 1开始，4结束，步长1，，，3开始，0结束，步长-1
# list_2 = list[1:4:]
# print(f"list列表 1开始，4结束，步长1:{list_2}")
# list_3 = list[3:0:-1]
# print(f"list列表 3开始，0结束，步长-1:{list_3}")
# #2.str字符串 头开始，尾结束，步长2 ，， 头开始，尾结束，步长-1
# str_2 = str[::2]
# print(f"str字符串 头开始，尾结束，步长2:{str_2}")
# str_3 = str[::-1]
# print(f"str字符串  头开始，尾结束，步长-1:{str_3}")
# #3.tuple元组 头开始尾结束，步长1，，，头5尾0，步长-2
# tuple_2 = tuple[::]
# print(f"tuple元组 头开始尾结束，步长1:{tuple_2}")
# tuple_3 = tuple[5::-2]
# print(f"tuple元组 头5尾0，步长-2:{tuple_3}")
# str_p = " fhfho,ahahaada,|aa子木小，evol，某杨|lllllallal"
# # str_p = str_p[::-1]
# # str_p = str_p.replace("aa","")
# # str_p = str_p.split("|")
# # print(str_p[1])
# a = str_p.index("杨")
# b = str_p.index("子")
# str_p = str_p[a:b-1:-1]
# print(str_p)

# my_list =['xiaoyang','小杨','xiaoyang',"小杨","xiaoli",'11',11,22,22]
# #定义一个空集合
# #通过for循环遍历列表
# #for循环中将列表元素添加至集合
# #集合打印输出
# my_set = set()
# for i in my_list:
#     my_set.add(i)
#     print(my_set)
# print(my_set)

# class Student:
#     def __init__(self,name,score,tel):
#         self.score=score
#         self.tel=tel
#         print("构建了一个类对象")
# stu = Student("小杨",99,1555)
# print(stu.tel)

# xinxi = {"小杨":{'部门':'科技部','工资':10000,'级别':5},
#          "小李":{'部门':'小杨部','工资':8000,'级别':1},
#          "linjunjie":{'部门':'suibian','工资':5000,'级别':1},
#          "junjie":{'部门':'xiaoli部','工资':7000,'级别':4},
#          "xiaomu":{'部门':'xiaoli部','工资':3000,'级别':1}}
# print(f"全体员工当前信息如下：\n{xinxi}")
# keys = xinxi.keys()
# for i in keys:
#     if xinxi[i]["级别"] == 1:
#         xinxi[i]["级别"] += 1
#         xinxi[i]["工资"] += 1000
#     else:
#         continue
# print(f"更新后\n{xinxi}")
#
# f = open("F:/周记/ai.txt","r",encoding="utf-8")
# print(type(f))
# print(f"{f.tell()}")
# print(f"读取前十个字符：{f.read(11)}")
# #print(f.seek(0))
# print(f"读取全部剩下字符{f.read()}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# str = input("Enter a number: ")
# nums = str.split(",")
# target = int(input("目标数："))
# i = 0
# while i <= len(nums)-1:
#     j = i + 1
#     while j <= len(nums)-1:
#         print(i,j)
#         print(nums[i],nums[j])
#         print(type(nums[j]))
#         if int(nums[i])+int(nums[j]) == target:
#             outs = [i,j]
#             print(outs)
#             i = len(nums)-1
#             break
#         j = j + 1
#     i = i + 1

# def func(list1,list2):#1
#     list3 = []
#     #print(type(list3))
#     for i in list1 :
#         if i in list2 :
#             list3.append(i)
#     return list3
# list1 = [1,2,3,4,5]
# list2 = [2,3,4,5,6]
# print(func(list1,list2))
# #2.字符串反串
# def func2(str1):
#     str1 = str1[::-1]
#     print(str1)
#     return str1
# str1 = "123456"
# print(func2(str1))
# #3统计莫个字符出现的次数
# def func3(str1,x):
#     count = str1.count(x)
#     return count
# str1 = "adajsnaldnald"
# x = input()
# count = func3(str1,x)
# print(count)
# #4.将每个单词首字母大写
# def func4(str1):
#     str2 = str1.title()
#     print(str2)
#     return str2
# str1 = "xiaoyang, i love you"
# print(func4(str1))
# #5.找到列表中第二大元素
# def func5(list1):
#     list1.sort()
#     print(list1)
#     list2 = list1[::-1]
#     print(list2)
#     return list1[-2]
# list1 = [1,12,2,3,4,5,6]
# print(func5(list1))
import json

# JSON 文件路径
json_file = r'F:\shujuji\annotations\instances_val2017.json'

# 检查映射关系是否正确
# 这里按你描述的规则
# helmet → 1, head → 2, person → 3
valid_ids = {1, 2, 3}

with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

errors = []
for ann in data['annotations']:
    cat_id = ann['category_id']
    if cat_id not in valid_ids:
        errors.append((ann['id'], cat_id))

if not errors:
    print("✅ 所有 category_id 都是合法值 (1, 2, 3)。")
else:
    print(f"❌ 发现 {len(errors)} 个非法 category_id:")
    for ann_id, cat_id in errors:
        print(f"  annotation id={ann_id}, category_id={cat_id}")
