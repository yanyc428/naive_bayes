# 基于朴素贝叶斯的文本分类

## To run

+ 确保拥有`python3.8`及以上的环境
+ 安装所需依赖`pip install -r requirements.txt`
+ 在`config.py`中配置数据，格式与`data`文件夹中保持一致。
+ 运行`python run.py -l -t 100 -e 10`
    + 参数`-l`，指定参数表示使用本地已经训练好的模型，否则重新计算相关概率
    + 参数`-t`，指定训练集大小，本项目已经提供了`10`和`100`规模的训练好的模型
    + 参数`-e`，指定测试集大小
    
## Contact Me
> 北京师范大学 政府管理学院 yanyuchen0428@163.com