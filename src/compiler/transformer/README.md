# Transformer

## 简介

转换器是整个编译栈的前端部分，其核心功能是将HNN编程框架描述的神经网络模型或者ONNX计算图转换成自定义的计算图IR，我们将其称为任务图，同时转换器也实现了一些计算图优化操作，例如算子融合、数据排布优化等（部分优化借助了onnx-optimizer和onnx-simplifier）。目前转换器可以支持HNN编程框架生成的带有量化信息的ANN的自动化转换和基于LIF神经元的SNN的自动化转换。

转换器的开发及使用细节请见工程文档。

## 基本使用

首先我们通过HNN编程框架定义一个神经网络：
```python
class LeNet(QModel):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = QConv2d(1, 6, 5, padding=2)
        self.maxpool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = QConv2d(6, 16, 5)
        self.maxpool2 = torch.nn.MaxPool2d(2, 2)
        self.flatten = Flatten3d()
        self.linear1 = QLinear(400, 120)
        self.linear2 = QLinear(120, 84)
        self.linear3 = QLinear(84, 10)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x
```
然后调用转换器的顶层接口来实现算法模型到任务图的转换：
```python
from src.compiler.transformer.transformer import PyTorchTransformer
from src.compiler.transformer.opt.optimize_config import OptimizeConfig


model = LeNet()
x = torch.randn(1, 1, 28, 28)  # dummy input
model_path = 'temp/lenet/lenet.pt'  # pretrained model
task_graph_path = 'temp/lenet/lenet.task'
opt = OptimizeConfig()
transformer = PyTorchTransformer(
    pytorch_model=model,
    task_graph_path=task_graph_path,
    case_name='lenet',
    optimize_config=opt,
    readable=True,
    pretrained_model_path=model_path
)
transformer.transform(input=x)  # generate IR file in task_graph_path
```