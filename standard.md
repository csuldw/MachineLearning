#Standard of Code
##变量
- 常量（全局变量）: 大写加下划线

>USER_CONSTANT

- 私有变量 : 小写和一个前导下划线

>_private_value

PS:保护变量

##函数和方法
- 私有方法 ： 小写和一个前导下划线

```
def _secrete(self):
    print "don't test me."
``` 
- 函数参数 : 小写和下划线，缺省值等号两边无空格

```
def connect(self, user=None):
    self._user = user
```
##类
- 类总是使用驼峰格式命名，即所有单词首字母大写其余字母小写。类名应该简明，精确，并足以从中理解类所完成的工作。常见的一个方法是使用表示其类型或者特性的后缀，例如:

```
SQLEngine
MimeTypes
```
- 对于基类而言，可以使用一个 Base 或者 Abstract 前缀

```
BaseCookie
AbstractGroup
```
##其他
- 使用 has 或 is 前缀命名布尔元素

```
is_connect = True
has_member = False
```
- 用复数形式命名序列

```
members = ['user_1', 'user_2']
```
- 用显式名称命名字典

```
person_address = {'user_1':'10 road WD', 'user_2' : '20 street huafu'}
```

- 避免通用名称

诸如 list, dict, sequence 或者 element 这样的名称应该避免。

* 一些数字

一行列数 : PEP 8 规定为 79 列，这有些苛刻了。根据自己的情况，比如不要超过满屏时编辑器的显示列数。这样就可以在不动水平游标的情况下，方便的查看代码。

一个函数 : 不要超过 30 行代码, 即可显示在一个屏幕类，可以不使用垂直游标即可看到整个函数。

一个类 : 不要超过 200 行代码，不要有超过 10 个方法。

一个模块 不要超过 500 行。

