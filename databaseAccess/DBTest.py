#!/bin/env python
# -*- encoding: utf-8 -*-
# -------------------------------------------------------------------------------
# Purpose:     example for python_to_mysql
# Author:      Silver Narcissus
# Created:     2017-06-28
# update:      2017-06-28
# -------------------------------------------------------------------------------
import os
import pymysql
#
# 建立和数据库系统的连接，格式
# conn   = MySQLdb.connect(host='localhost',user='root',passwd='123456',db='test',port=3306,charset='utf8')

# 指定配置文件，确定目录,或则写绝对路径
cwd = os.path.realpath(os.path.dirname(__file__))
db_conf = os.path.join(cwd, 'db.conf')
conn = pymysql.connect(read_default_file=db_conf, host='localhost', db='python', port=3306, charset='utf8')

# 要执行的sql语句
query = "select * from news"

# 获取操作游标
cursor = conn.cursor()

# 执行SQL
cursor.execute(query)

# 获取一条记录,每条记录做为一个元组返回,返回3，游标指到第2条记录。
result1 = cursor.fetchone()
for i in result1:
    print(i)
