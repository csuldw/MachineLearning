## 最小冗余最大相关(minimal Redundancy Maximal Relevance,mRMR)

mRMR方法是用来做特征选择的，该方法保证了特征间的最小冗余性以及特征和类标签的最大相关性，详细的源码请前往原链接[mRMR](http://penglab.janelia.org/proj/mRMR/)，本文主要介绍如何使用此方法。关于本软件的数学公式推导，可以参考这篇paper，[Pent et al. mRMR.pdf](https://github.com/csuldw/MachineLearning/tree/master/mRMR/Pent et al_mRMR.pdf).

在这里，主要介绍如何将该方法运行起来。

## 使用方法

在Linux环境下将mrmr_c_src.zip文件直接解压后就可以使用了，无需安装。

```
[root@master mrmr_c_src]# ./mrmr

Usage: mrmr_osx -i <dataset> -t <threshold> [optional arguments]
         -i <dataset>    .CSV file containing M rows and N columns, row - sample, column - variable/attribute.
         -t <threshold> a float number of the discretization threshold; non-spec ifying this parameter means no discretizaton (i.e. data is already integer); 0 to make binarization.
         -n <number of features>   a natural number, default is 50.
         -m <selection method>    either "MID" or "MIQ" (Capital case), default is MID.
         -s <MAX number of samples>   a natural number, default is 1000. Note that if you don't have or don't need big memory, set this value small, as this program will use this value to pre-allocate memory in data file reading.
         -v <MAX number of variables/attibutes in data>   a natural number, default is 10000. Note that if you don't have or don't need big memory, set this value small, as this program will use this value to pre-allocate memory in data file reading.
         [-h] print this message.


 *** This program and the respective minimum Redundancy Maximum Relevance (mRMR)               
     algorithm were developed by Hanchuan Peng <hanchuan.peng@gmail.com>for
     the paper
     "Feature selection based on mutual information: criteria of
      max-dependency, max-relevance, and min-redundancy,"
      Hanchuan Peng, Fuhui Long, and Chris Ding,
      IEEE Transactions on Pattern Analysis and Machine Intelligence,
      Vol. 27, No. 8, pp.1226-1238, 2005.
```

在Linux环境下使用nohup提交下列命令

参数说明见上


```
[root@master mrmr_c_src]# nohup ./mrmr -i /home/liudiwei/data/featselect/combine_f195.data -t 0.0001 -s 1000000 -v 1000 -n 42 > result.out  2>&1 &
```


> 警示：由于数据太大，就不提供完整的数据集，trainset.data中只是元数据集的Top10,作为数据格式参照。


********************************************************************************

Result:

最后会将文件输出值result.out中

- result.out文件
