# -*- coding:utf-8 -*-
# @Author  :Wan Linan
# @time    :2021/12/15 16:42
# @File    :船属性图谱.py
# @Software:PyCharm
"""
@remarks :
"""
import pandas
from py2neo import Graph, Node, Relationship
import random
g = Graph(
    "http://localhost:7474/",
    username="neo4j",
    password="wan1997linan"
)
g.run("match (n) detach delete n")


def Createkg():
    frame = pandas.read_excel(r"D:\HUST\prcharm_env_pytorch\shipKG\ship.xlsx")
    n = len(frame)
    print(frame)
    ship2class = {}

    # 创建船类型节点
    shipzl = list(set(frame['船类型']))
    m = len(shipzl)
    for i in range(m):
        g.run("MERGE(p: Class{Name: '%s'})" % (shipzl[i]))

    for i in range(n):
        ship2class[frame['船名'][i]] = frame['船类型'][i]
        # 创建船舶节点
        g.run("MERGE(p: Ship{Name: '%s',weight: '%s',length: '%s',width: '%s'})" %
              (frame['船名'][i], frame['净吨'][i], frame['船舶总长'][i], frame['型宽'][i]))
        g.run("MERGE(p: Weight{Name: '%s'})" % (frame['净吨'][i]))
        g.run("MERGE(p: Length{Name: '%s'})" % (frame['船舶总长'][i]))
        g.run("MERGE(p: Width{Name: '%s'})" % (frame['型宽'][i]))
        # 创建船舶和类型之间的关系
        g.run(
            "MATCH(e: Ship), (cc: Class) \
            WHERE e.Name='%s' AND cc.Name='%s'\
            CREATE(e)-[r:%s{relation: '%s'}]->(cc)\
            RETURN r" % (frame['船名'][i], frame['船类型'][i], 'Type', '种类')
        )
        g.run(
            "MATCH(e: Ship), (cc: Weight) \
            WHERE e.Name='%s' AND cc.Name='%s'\
            CREATE(e)-[r:%s{relation: '%s'}]->(cc)\
            RETURN r" % (frame['船名'][i], frame['净吨'][i], 'Weight', '种类')
        )
        g.run(
            "MATCH(e: Ship), (cc: Length) \
            WHERE e.Name='%s' AND cc.Name='%s'\
            CREATE(e)-[r:%s{relation: '%s'}]->(cc)\
            RETURN r" % (frame['船名'][i], frame['船舶总长'][i], 'Length', '种类')
        )
        g.run(
            "MATCH(e: Ship), (cc: Width) \
            WHERE e.Name='%s' AND cc.Name='%s'\
            CREATE(e)-[r:%s{relation: '%s'}]->(cc)\
            RETURN r" % (frame['船名'][i], frame['型宽'][i], 'Width', '种类')
        )


Createkg()