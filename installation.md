---
layout: chapter
title: 安装
---

为顺利帮助读者完成本书中实例的学习，译者已对代码做了相应整理，下面给出在对本书实例学习前，你需要做的前期安装工作。注意，下面译者给出的安装过程是针对**Windows**下的，对于本书的PCV库，在别的平台上如Linux和Mac下是一样的；而对于一些实例中依赖的库，比如matplotlib、pylab、OpenCV等，在别的平台上，请按相应库文档安装说明安装。

<h2 id="sec-0-1">0.1 需要准备的安装包</h2>

要完整复现书中的实例，你需要的主要三个文件包括Python(x,y) 2.7.x安装包、PCV库和本书用到的数据库。Python(x,y)可以在[Google Code]((https://code.google.com/p/pythonxy/)),PCV库、本书整理出来的实例代码以及本书用到的所有图像数据可以从[首页](http://yuanyong.org/pcvwithpython/)给出的链接下载。

<h2 id="sec-0-2">0.2 安装Python(x,y)</h2>

在Windows下，译者推荐你安装Python(x,y) 2.7.x。Python(x,y) 2.7.x是一个库安装包，除了包含Python自身外，还包含了很多第三方库，下面是安装Python(x,y)时的界面：
![ch02_fig2-1_harris](assets/images/figures/pre/pythonxy01.jpg)
![ch02_fig2-1_harris](assets/images/figures/pre/pythonxy02.png)
从上面第二幅图可以看出，pythonxy不仅包含了SciPy、NumPy、PyLab、OpenCV、MatplotLib,还包含了机器学习库scikits-learn。
为避免出现运行实例时出现的依赖问题，译者建议将上面的库全部选上，也就是选择“full”(译者也是用的全部安装的方式进行后面的实验的)。安装完成后，为验证安装是否正确，可以在Python shell里确认一下OpenCV是否已安装来进行验证，在Python Shell里输入下面命令：

```python
from cv2 import __version__
__version__
```
输入上面命令，如果可以看到OpenCV的版本信息，则说明python(x,y)已安装正确。

另外，需要提醒读者的是，Python是没有平台区分的，这里指的平台不是指Linux和Mac这样的平台概念，而是在Windows上没有位数的区分。举个简单的例子，比如你是64位的Windows系统，你可以安装32位的Python。对于这一部分的详细说明，可以参阅译者的一篇博文[Django配置MySQL](http://yuanyong.org/blog/config-mysql-for-django.html)最后一段的说明。好了，关于Python(x,y)的安装说明就说到这里。

<h2 id="sec-0-3">0.3 安装PCV库</h2>

衷心的祝福你们，

<p class="align-right">《Ruby on Rails Tutorial》作者 Michael Hartl</p>


Best wishes and good luck,

<p class="align-right">Michael Hartl<br />
Author<br />
The Ruby on Rails Tutorial</p>

<div class="navigation">
  <a class="prev_page" href="author.html">&laquo; 作者译者</a>
  <a class="next_page" href="foreword.html">序 &raquo;</a>
</div>
