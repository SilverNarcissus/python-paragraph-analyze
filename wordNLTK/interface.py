import os
import pickle
#
# 建立和数据库系统的连接，格式
# conn   = MySQLdb.connect(host='localhost',user='root',passwd='123456',db='test',port=3306,charset='utf8')
# 指定配置文件，确定目录,或则写绝对路径
import time

import pymysql

from wordNLTK.analyze import bigram_word_feats, preprocess


def process():
    cwd = os.path.realpath(os.path.dirname(__file__))
    db_conf = os.path.join(cwd, 'db.conf')
    conn = pymysql.connect(read_default_file=db_conf, host='localhost', db='python', port=3306, charset='utf8')

    # 要执行的sql语句
    # noinspection SqlResolve
    select_query = "SELECT id, content FROM news"
    # noinspection SqlResolve


    # 获取操作游标
    select_cursor = conn.cursor()
    update_cursor = conn.cursor()

    # 执行SQL
    select_cursor.execute(select_query)

    # 导入分类器
    classifier = pickle.load(open('/Users/SilverNarcissus/PycharmProjects/wordAnalysis/reviews_classifier.pkl', 'rb'))

    # print(select_cursor.fetchone())
    # 获取一条记录,每条记录做为一个元组返回
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for row in select_cursor.fetchall():
        news_id = row[0]
        news_content = row[1]
        news_emotion = classifier.classify(bigram_word_feats(preprocess(news_content)))
        update_cursor.execute("UPDATE news SET emotion = '" + news_emotion + "' WHERE id = " + str(news_id))

    conn.commit()
    print("here")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
