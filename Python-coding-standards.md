# Python-Coding-Standards

## 标准头部

```python
#!/usr/bin/python
#-*- coding:utf8 -*- 
"""
description
"""
import sys
import os
```

## 常量

常量名所有字母大写，由下划线连接各个单词，如：

```
USER_CONSTANT
```

## 变量 

变量名全部小写，由下划线连接各个单词，如：

```
color = WHITE
this_is_a_variable = 1
```

## 函数和方法

- 私有方法 ： 小写和一个前导下划线

```python
def _secrete(self):
    print "don't test me."
``` 

- 函数参数 : 小写和下划线，缺省值等号两边无空格


```python
def connect(self, user=None):
    self._user = user
```


## 类

- 类总是使用驼峰格式命名，不使用下划线连接单词，也不加入 C、T 等前缀，即所有单词首字母大写其余字母小写。类名应该简明，精确，并足以从中理解类所完成的工作。常见的一个方法是使用表示其类型或者特性的后缀，例如:

```python
class SQLEngine(object):
    pass
```

- 对于基类而言，可以使用一个 Base 或者 Abstract 前缀

```python
class BaseCookie(object):
	pass
class AbstractGroup(object):
	psss
```

## 特定命名方式
主要是指 \_\_xxx__ 形式的系统保留字命名法。项目中也可以使用这种命名，它的意义在于这种形式的变量是只读的，这种形式的类成员函数尽量不要重载。如


```python
class Base(object):
    def __init__(self, id, parent = None):
        self.__id__ = id
        self.__parent__ = parent
    def __message__(self, msgid):
        # ...略
```

其中 \_id、parent_ 和 \_message_ 都采用了系统保留字命名法。

## 空格

1. 在二元算术、逻辑运算符前后加空格：如 a = b + c；
- 在一元前缀运算符后不加空格，如 if !flg: pass；
- ":"用在行尾时前后皆不加空格，如分支、循环、函数和类定义语言；用在非行尾时后端加空格，如 dict 对象的定义 d = {'key': 'value'};
- 括号（含圆括号、方括号和花括号）前后不加空格，如 do\_something(arg1, arg2), 而不是 do_something( arg1, arg2 )；
- 逗号后面加一个空格，前面不加空格。

## 空行

1. 在类、函数的定义间加空行；
- 在import不同种类的模块间加空行；
- 在函数中的逻辑段落间加空行，即把相关的代码紧凑写在一起，作为一个逻辑段落，段落间以空行分隔。

## 断行

（1）行的最大长度不得超过 80 个字符的标准。折叠长行的方法有以下几种方法：
1）为长变量名换一个短名，如：

```python
this.is.a.very.long.variable_name = this.is.another.long.variable_name
```

应改为：

```python
variable_name1 = this.is.a.very.long.variable_name
variable_name2 = this.is.another.variable_name
variable_name1 = variable_name2
```

（2）在括号（包括圆括号、方括号和花括号）内换行，如：

```python
class Edit(CBase):
    def __init__(self, parent, width,
                font = FONT, color = BLACK, pos = POS, style = 0):
```

或：

```python
very_very_very_long_variable_name = Edit(parent, \
                                         width, \
                                         font, \
                                         color, \
                                         pos)
```

（3）在长行加入续行符强行断行，断行的位置应在操作符前，且换行后多一个缩进，以使维护人员看代码的时候看到代码行首即可判定这里存在换行，如：

```python
if color == WHITE or color == BLACK \
              or color == BLUE:
	do_something(color);
```



## 语句

- import

import 语句有以下几个原则需要遵守：

（1）import 的次序，先import Python内置模块，再import第三方模块，最后import自己开发的项目中的其它模块；这几种模块中用空行分隔开来。

（2）一条import语句import一个模块。

（3）当从模块中 import 多个对象且超过一行时，使用如下断行法（此语法 py2.5 以上版本才支持）：

```python
from module import (obj1, obj2, obj3, obj4,
obj5, obj6)
```
4）不要使用 `from module import *`，除非是 `import `常量定义模块或其它你确保不会出现命名空间冲突的模块。


## 赋值

对于赋值语言，等号前后空一格，格式如下：

```python
a = 1
variable = 2
fn = callback_function
```

## 分支和循环

不要写成一行，如：

```python
if !flg: pass
for i in xrange(10): print i
```

应该写成：

```python
if !flg:
    pass
for i in xrange(10):
    print i
```
## 其他

- 使用 has 或 is 前缀命名布尔元素

```python
is_connect = True
has_member = False
```
- 用复数形式命名序列

```python
members = ['user_1', 'user_2']
```
- 用显式名称命名字典

```python
person_address = {'user_1':'10 road WD', 'user_2' : '20 street huafu'}
```

- 避免通用名称

诸如 list, dict, sequence 或者 element 这样的名称应该避免。

* 一些数字

一行列数 : PEP 8 规定为 79 列，这有些苛刻了。根据自己的情况，比如不要超过满屏时编辑器的显示列数。这样就可以在不动水平游标的情况下，方便的查看代码。

一个函数 : 不要超过 30 行代码, 即可显示在一个屏幕类，可以不使用垂直游标即可看到整个函数。

一个类 : 不要超过 200 行代码，不要有超过 10 个方法。

一个模块 不要超过 500 行。

## Contributor

- Liu Diwei: https://github.com/csuldw
- Gao Yong: https://github.com/gaoyongcn
