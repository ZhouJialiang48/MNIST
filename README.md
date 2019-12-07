
### 文件说明

1. forward.py：模型结构脚本
2. backward.py：模型训练脚本，用于训练模型参数
3. evaluate.py：模型评估脚本，用于选择最佳模型
4. predict.py：模型预测脚本，用于测试模型在新数据上的表现
5. img/：新数据的目录，现有一张图片，可自行添加，验证模型对其它图片的预测效果



### 操作步骤

##### 1. 获取项目文件，切换到工作目录下
```bash
git clone git@github.com:zhoujl48/MNIST.git
cd MNIST/
```

##### 2. 模型训练和模型选择，此时生成两个目录，`data/`和`model/`，分别存放数据集和训练好的模型
```bash
python backwards.py
python evaluation.py
```

##### 3. 输入新的图片，验证模型效果
```bash
python predict.py
```
