# -*- coding:utf-8 -*-
# @Author  :Wan Linan
# @time    :2021/4/15 10:27
# @File    :ship150.py
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

    for i in range(n):
        ship2class[frame['船名'][i]] = frame['船类型'][i]
        # 创建船舶节点
        g.run("MERGE(p: %s{Name: '%s',weight: '%s',length: '%s',width: '%s'})" %
              (frame['船类型'][i], frame['船名'][i], frame['净吨'][i], frame['船舶总长'][i], frame['型宽'][i]))

    # 创建船舶关系，协同、跟随、躲避，没船类型节点
    li1 = ['No.' for _ in range(80)]
    li2 = [li1[i] + str(i + 1) for i in range(80)]
    for i in range(20):
        li3 = random.sample(li2, 2)
        g.run(
            "MATCH(e: %s), (cc: %s) \
            WHERE e.Name='%s' AND cc.Name='%s'\
            CREATE(e)-[r:%s{relation: '%s'}]->(cc)\
            RETURN r" % (ship2class[li3[0]], ship2class[li3[1]], li3[0], li3[1], 'cooperate', '船间关系')
        )
    for i in range(20):
        li3 = random.sample(li2, 2)
        g.run(
            "MATCH(e: %s), (cc: %s) \
            WHERE e.Name='%s' AND cc.Name='%s'\
            CREATE(e)-[r:%s{relation: '%s'}]->(cc)\
            RETURN r" % (ship2class[li3[0]], ship2class[li3[1]], li3[0], li3[1], 'follow', '船间关系')
        )
    for i in range(20):
        li3 = random.sample(li2, 2)
        g.run(
            "MATCH(e: %s), (cc: %s) \
            WHERE e.Name='%s' AND cc.Name='%s'\
            CREATE(e)-[r:%s{relation: '%s'}]->(cc)\
            RETURN r" % (ship2class[li3[0]], ship2class[li3[1]], li3[0], li3[1], 'avoid', '船间关系')
        )


Createkg()
