---
layout: chapter
title: 第七章 图像搜索
---

本章将展示如何利用文本挖掘技术基于图像视觉内容进行图像搜索。在本章中，阐明了利用视觉单词的基本思想，完整解释了的安装细节，并且还在一个示例数据集上进行测试。

本章图像搜索模型是建立在BoW词袋基础上，先对图像数据库提取sift特征，对提取出来的所有sift特征进行kmeans聚类得到视觉单词(每个视觉单词用逆文档词频配以一定的权重)，然后对每幅图像的sift描述子进行统计得到每幅图像的单词直方图表示，最后对给定的查询图像，将其对应的单词直方图与数据库中的单词直方图进行欧式距离匹配，并由大到小进行排序，最后显示靠前的图像。

<h2 id="sec-7-1">7.1 创建词汇</h2>

为创建视觉单词词汇，首先需要提取特征描述子，这里，我们使用SIFT描述子。

```python
# -*- coding: utf-8 -*-
import pickle
from PCV.imagesearch import vocabulary
from PCV.tools.imtools import get_imlist
from PCV.localdescriptors import sift

#获取图像列表
imlist = get_imlist('./first500/')
nbr_images = len(imlist)
#获取特征列表
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]

#提取文件夹下图像的sift特征
for i in range(nbr_images):
    sift.process_image(imlist[i], featlist[i])

#生成词汇
voc = vocabulary.Vocabulary('ukbenchtest')
voc.train(featlist, 1000, 10)
#保存词汇
# saving vocabulary
with open('./first500/vocabulary.pkl', 'wb') as f:
    pickle.dump(voc, f)
print 'vocabulary is:', voc.name, voc.nbr_words
```
上面源码对应[ch07_cocabulary.py](https://github.com/willard-yuan/pcv-book-code/tree/master/ch07)。在该文件夹下，有一个first500的文件夹，将你从首页下载的[数据](http://yuanyong.org/pcvwithpython/)中文件夹first1000中的图像放在first500中。注意，译者这里实验的时候，由于计算机内存不足，所以只从first1000取出前500张放入first500中。

运行上面代码，会在first500文件夹下生成一个名为vocabulary.pkl的文件，同时在first500会多出500个后缀为.sift的文件，它们分别对应每幅图像提取出来的sift特征描述子。

<h2 id="sec-7-2">7.2 添加图像</h2>

```python
# -*- coding: utf-8 -*-
import pickle
from PCV.imagesearch import imagesearch
from PCV.localdescriptors import sift
from sqlite3 import dbapi2 as sqlite
from PCV.tools.imtools import get_imlist

#获取图像列表
imlist = get_imlist('./first500/')
nbr_images = len(imlist)
#获取特征列表
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]

# load vocabulary
#载入词汇
with open('./first500/vocabulary.pkl', 'rb') as f:
    voc = pickle.load(f)
#创建索引
indx = imagesearch.Indexer('testImaAdd.db',voc)
indx.create_tables()
# go through all images, project features on vocabulary and insert
#遍历所有的图像，并将它们的特征投影到词汇上
for i in range(nbr_images)[:500]:
    locs,descr = sift.read_features_from_file(featlist[i])
    indx.add_to_index(imlist[i],descr)
# commit to database
#提交到数据库
indx.db_commit()

con = sqlite.connect('testImaAdd.db')
print con.execute('select count (filename) from imlist').fetchone()
print con.execute('select * from imlist').fetchone()
```
运行上面代码后，会在根目录生成建立的索引数据库testImaAdd.db，

<h2 id="sec-7-3">7.3 获取候选图像</h2>

```python
# -*- coding: utf-8 -*-
import pickle
from PCV.imagesearch import imagesearch
from PCV.localdescriptors import sift
from sqlite3 import dbapi2 as sqlite
from PCV.tools.imtools import get_imlist

#获取图像列表
imlist = get_imlist('./first500/')
nbr_images = len(imlist)
#获取特征列表
featlist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]

#载入词汇
f = open('./first500/vocabulary.pkl', 'rb')
voc = pickle.load(f)
f.close()

src = imagesearch.Searcher('testImaAdd.db',voc)
locs,descr = sift.read_features_from_file(featlist[0])
iw = voc.project(descr)

print 'ask using a histogram...'
#获取imlist[0]的前十幅候选图像
print src.candidates_from_histogram(iw)[:10]

src = imagesearch.Searcher('testImaAdd.db',voc)
print 'try a query...'

nbr_results = 12
res = [w[1] for w in src.query(imlist[0])[:nbr_results]]
imagesearch.plot_results(src,res)
```

<h2 id="sec-7-4">7.4　建立演示程序及Web应用</h2>

```python
# -*- coding: utf-8 -*-
import cherrypy
import pickle
import urllib
import os
from numpy import *
#from PCV.tools.imtools import get_imlist
from PCV.imagesearch import imagesearch

"""
This is the image search demo in Section 7.6.
"""


class SearchDemo:

    def __init__(self):
        # 载入图像列表
        self.path = './first500/'
        #self.path = 'D:/python_web/isoutu/first500/'
        self.imlist = [os.path.join(self.path,f) for f in os.listdir(self.path) if f.endswith('.jpg')]
        #self.imlist = get_imlist('./first500/')
        #self.imlist = get_imlist('E:/python/isoutu/first500/')
        self.nbr_images = len(self.imlist)
        self.ndx = range(self.nbr_images)

        # 载入词汇
        f = open('./first500/vocabulary.pkl', 'rb')
        self.voc = pickle.load(f)
        f.close()

        # 显示搜索返回的图像数
        self.maxres = 49

        # header and footer html
        self.header = """
            <!doctype html>
            <head>
            <title>Image search</title>
            </head>
            <body>
            """
        self.footer = """
            </body>
            </html>
            """

    def index(self, query=None):
        self.src = imagesearch.Searcher('testImaAdd.db', self.voc)

        html = self.header
        html += """
            <br />
            Click an image to search. <a href='?query='> Random selection </a> of images.
            <br /><br />
            """
        if query:
            # query the database and get top images
            #查询数据库，并获取前面的图像
            res = self.src.query(query)[:self.maxres]
            for dist, ndx in res:
                imname = self.src.get_filename(ndx)
                html += "<a href='?query="+imname+"'>"
                html += "<img src='"+imname+"' width='200' />"
                html += "</a>"
            # show random selection if no query
            # 如果没有查询图像则随机显示一些图像
        else:
            random.shuffle(self.ndx)
            for i in self.ndx[:self.maxres]:
                imname = self.imlist[i]
                html += "<a href='?query="+imname+"'>"
                html += "<img src='"+imname+"' width='200' />"
                html += "</a>"

        html += self.footer
        return html

    index.exposed = True

#conf_path = os.path.dirname(os.path.abspath(__file__))
#conf_path = os.path.join(conf_path, "service.conf")
#cherrypy.config.update(conf_path)
#cherrypy.quickstart(SearchDemo())

cherrypy.quickstart(SearchDemo(), '/', config=os.path.join(os.path.dirname(__file__), 'service.conf'))
```
<h2 id="sec-7-5">7.5　配置service.conf</h2>

```sh
[global]
server.socket_host = "127.0.0.1"
server.socket_port = 8080
server.thread_pool = 10
tools.sessions.on = True
[/]
tools.staticdir.root = "E:/python/isoutu"
[/first500]
tools.staticdir.on = True
tools.staticdir.dir = "first500"
```
