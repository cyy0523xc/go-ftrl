# go-ftrl
FTRL-Proximal for golang. From Google's paper: Ad Click Prediction: a View from the Trenches

## Install

```sh
go get -U github.com/cyy0523xc/go-ftrl
```

## 测试数据

见：./libsvm_data.txt

格式采用libsvm算法所使用的格式，如：

```
1 16:1 19:1 24:1 26:1 58:1 682:1 733:1 764:1 778:1 788:1 1247:1 1251:1 1256:1 1258:1
0 16:1 19:1 24:1 45:1 262:1 682:1 733:1 764:1 778:1 788:1 1247:1 1251:1 1256:1 1258:1
0 16:1 19:1 24:1 48:1 353:1 682:1 733:1 764:1 778:1 788:1 1247:1 1251:1 1256:1 1258:1
0 16:1 19:1 24:1 43:1 224:1 682:1 733:1 764:1 778:1 788:1 1247:1 1251:1 1256:1 1258:1
```

说明:

- 其中第一列的0,1是值，例如广告点击则为1, 不点击则为0。
- 后面的`index:feature_val`, 冒号前面的为特征序号，后面的则为特征值
- 特征之间有且只有一个空格

## 训练与测试部分

```

```


## 预测部分


