#!/usr/bin/env python
#coding:utf-8

container = {}
# container = {
#     # "第一个人的随机字符串":{}，
#     # "第一个人的随机字符串":{'k1': 111, 'parents': '你'}，
# }

class Session:
    def __init__(self, handler,gqshijian):
        """
        创建Session()对象时接收两个参数
        参数1、接收RequestHandler的self对象，也就是继承RequestHandler对象
        参数2、接收cookie过期时间天数
        """
        self.handler = handler                                  #handler接收tornado.web.RequestHandler的self对象，也就是继承RequestHandler对象
        self.random_str = None                                  #初始化随机密串
        self.gqshijian = gqshijian                              #获取设置cookie过期时间

    def __genarate_random_str(self):                            #生成随机密串
        import hashlib                                          #导入md5加密模块
        import time                                             #导入时间模块
        obj = hashlib.md5()                                     #创建md5加密对象
        obj.update(bytes(str(time.time()), encoding='utf-8'))   #获取系统当前时间，进行md5加密
        random_str = obj.hexdigest()                            #获取加密后的md5密串
        return random_str                                       #返回加密后的md5密串

    def __setitem__(self, key,value):                                   #当创建Session对象，后面跟着[xxx]=xxx的时候自动执行并且接收[xxx]=xxx的值
        """
        使用方法：Session对象[xxx]=xxx
        功能：随机生成密串写入cookie，接收自定义用户数据，添加到cookie密串对应的字典里
        """
        # 在container中加入随机字符串
        # 定义专属于自己的数据
        # 在客户端中写入随机字符串
        # 判断，请求的用户是否已有随机字符串
        if not self.random_str:                                         #判断初始化密串不存在
            random_str = self.handler.get_cookie('xr_cookie')          #获取客服端浏览器get_cookie里的密串
            if not random_str:                                          #判断客服端浏览器get_cookie里如果没有密串
                random_str = self.__genarate_random_str()               #执行密串生成方法，生成密串
                container[random_str] = {}                              #在container字典里添加一个，密串作为键，值为空字典的元素
            else:                                                       #如果客服端浏览器get_cookie里有密串
                # 客户端有随机字符串
                if random_str in container.keys():                      #判断密串在container字典的键里是否存在
                    pass                                                #如果存在什么都不做
                else:                                                   #如果不存在
                    random_str = self.__genarate_random_str()           #重新生成密串
                    container[random_str] = {}                          #在container字典里添加一个，密串作为键，值为空字典的元素
            self.random_str = random_str                                #将密串赋值给，初始化密串
        # 如果用户密串初始化就存在，说明登录过并且cookie和container字典里都存在
        container[self.random_str][key] = value                                                     #找到container字典里键为密串的元素值是一个字典，将接收到的key作为键，value作为值添加到元素字典里
        self.handler.set_cookie("xr_cookie", self.random_str,expires_days = self.gqshijian)        #将密串作为cookie值，向浏览器写入cookie

    def __getitem__(self,key):                                          #当创建Session对象，后面跟着[xxx]自动执行，并接收[xxx]的值
        """
        使用方法：Session对象[xxx]
        功能：获取cookie对应字典里，键为接收到参数的值，存在返回值，不存在返回None
        """
        # 获取客户端的随机字符串
        # 从container中获取专属于我的数据
        #  专属信息【key】
        random_str =  self.handler.get_cookie("xr_cookie")             #获取cookie里的密串
        if not random_str:                                              #判断cookie里的密串如果不存在
            return None                                                 #返回None
        # 客户端有随机字符串
        user_info_dict = container.get(random_str,None)                 #在container字典里找到密串对应的元素
        if not user_info_dict:                                          #如果container字典里没有密串对应的元素
            return None                                                 #返回None
        #如果cookie里的密串存在，并且container字典里也存在密串对应的元素
        value = user_info_dict.get(key, None)                           #接收用户传来的值，将值作为键找到字典里对应的值
        return value                                                    #返回值