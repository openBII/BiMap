
# BiMap: Brain-inspired Many-core Architecture exploration Platform

类脑众核架构探索平台（Brain-inspired Many-core Architecture exploration Platform, Bi-Map），是旨在提供灵活通用的类脑部署解决方案、进行类脑体系结构关键技术探索的类脑系统软件，为编译与芯片架构设计这两大类脑系统中不可分割的研究方向提供服务。我们希望通过Bi-Map，使用者可以快速高效的将深度学习模型与传统类脑计算模型部署到类脑芯片上，也可以通过编译与仿真的协同优化，探索更好的编译映射策略与类脑架构设计。

BiMap计划包含如下组件：

![image](docs/sphinx/source/总述/_static/framework2.png)

**编译栈：** 将算法模型转换为芯片可执行的代码。包括，

- **I1, NN IR (Neural Network Intermediate Representation)：** 统一不同深度学习框架（如PyTorch、TensorFlow等）编写的算法模型的表示，其解耦了计算系统与上层各种各样的编程框架。当前，Bi-Map使用 `ONNX` 作为NN IR。
- **I2, Task IR ：** 利用功能性原语表达算法执行任务的IR，其满足类脑系统的功能与精度约束，表达了类脑系统原生的支持范围（如是否支持SNN、是否支持训练、是否支持浮点计算等）。Task IR解耦了计算系统的功能与硬件的众核结构设计。 
- **C1, 转换器 ：** 将NN IR转换为Task IR，主要包括等价转换：将ONNX转换为Task IR功能性原语，构造Task IR图结构；非等价转换：将高精度计算量化为低精度计算、使用神经网络或查找表等通用逼近算子逼近非线性操作或非原生支持的操作。
- **I3, Mapping IR ：** 表示任务在众核架构上的时空分布的IR，由经过图变换的任务图（Task IR）和任务图与硬件资源的时空映射关系组成，解耦了硬件的众核结构设计与具体硬件组件的实现。
- **C2, 映射器 ：** 将Task IR转换为Mapping IR，包括任务的空间分配：每个核、每个存储计算组件执行什么任务；时间调度：每个核在什么时间执行什么任务；组织映射策略搜索。
- **I3, Assembly IR ：** 表示任务在每个硬件组件（计算组件、内存组件、片上网络、控制组件）上执行行为与逻辑的IR，大致相当于传统计算机的汇编程序，其解耦了硬件行为与硬件RTL级实现。
- **C3, 代码生成器 ：** 将Mapping IR转换为Assembly IR，根据硬件的执行模型，进行内存地址分配、路由优化、指令参数生成等等步骤。
- **I5, Code IR ：** 硬件的可执行代码或硬件可直接解析的编码。
- **C4, 汇编器 ：** 将Assembly IR转换为Code IR。汇编器同样可以包含链接过程，即将多个芯片或任务的独立编译结果链接起来。

**仿真（执行）栈：** 仿真栈在不同的抽象层次上表达硬件的功能与结构设计，实现执行IR、仿真硬件、评估性能等功能。在探索层面，Bi-Map目前侧重于编译探索部分，仿真栈以天机X芯片为基础构建。包括，

- **S1, HNN编程框架 ：** 编写ANN、SNN及二者混的算法模型的编程框架。如果编写ANN, 可以直接使用PyTorch、TensorFlow等成熟的深度学习编程框架。（编程框架可以不看作是仿真栈的一部分）
- **S2, Task Rabbit ：** 执行Task IR的程序，相当于硬件的支持功能的模板，不暴露除功能外的其他硬件信息。其实现上是一个简单的图执行引擎，会输出Task IR在给定输入上的运行结果。我们给它起名叫Task Rabbit。
- **S3, 性能级仿真器 ：** 评估一个映射策略的资源占用（执行时间、内存量、通信量等）的静态仿真器。同时也是一个可交互仿真器，其暴露一系列映射动作与评估接口，供映射策略搜索使用。
- **S4, 行为级仿真器 ：** 仿真硬件计算、控制、内存、路由行为的仿真器，会输出与硬件等价的运行中间结果。其建模了硬件各种组件的执行机制。
- **S5, CModel（时钟精确级仿真器）：** 逐时钟的仿真硬件的执行，包含硬件的时序信息，相当于RTL逻辑。

**测试栈：** 测试栈管理测试用例，组织测试流程，是各个软件模块如期工作的保障，并可以看作类脑体系结构开发的数据集。包括：

- **测试用例库：** 存储各个层次的IR作为测试用例。
- **结果比对框架：** 对比1、2、4、5结果输出点，判断执行结果是否正确。
- **性能评估框架：** 对比2、3、4、5运行效果输出点，分析性能表现。

**运行时** 在芯片执行期间运行的系统协作软件。包括：

- **R1, IO Streamer ：** 在运行时将外部输入数据打入众核上的特定核；收集众核中特定核的数据到外部输出。
- **R2, 数据处理组件：** 进行数据输入之前的处理，如分块、维度变换等；对处理结果进行处理得到外部输出数据，如合并、维度变换等。
- **R3, trigger生成器：** 生成控制芯片运行的各种触发信号。 

**其他辅助工具** 辅助工程开发或者系统执行的软件工具。包括：

- **O1, IR转换工具：** 因为一些历史原因，IR设计发生过数次版本迭代，为了其余组件正常工作，IR转换工具负责将转换不同的IR设计。（本开源工程不涉及）
- **O2, 执行器：** 封装了各组件执行的高层逻辑，并对外提供使用整个系统的接口。
- **O3, 比较器：** 封装了组件间输出结果的比较逻辑，用于测试框架的正确性判定。
- **O4, 可视化工具：** 可视化某些IR实例和部分组件执行中间状态，主要用于调试。

详细文档见 `docs/spnix`

## IR 文件格式统一

+ Task IR为转换层的输出，使用 `.task`文件后缀
+ Mapping IR为映射层的输出，使用 `.map`文件后缀
+ Assembly IR为代码生成层的输出，使用 `.asm`文件后缀

## 执行

```bash
pip install -r requirement.txt
source setenv.sh
```

#### behavior Simulator (C++)

**编译**

```bash
cd build
make -f behavior_simulator.mak -j
```

**执行**

命令行指令, 例子：

```
./build/behavior_simulator -i ${TEST_LIB}/behavior_lib/1C1P/P02/P02001.asm.txt -c P02001
```

VSCode执行：

在launch.json中设置输入参数，在Run and Debug活动面板中选择Launch Behavior Simulator并执行。

Executor执行：

```
python flow/executors/exe_behavior_simulator.py
```

**输入**

* -i：必选参数，输入配置文件路径
* -c：必选参数，用例名
* -o：可选参数，输出结果路径名，默认temp/${CASE_NAME}/behavior_out
* -r：可选参数，是否输出可读的形式，默认为true

**输出**

* temp/${CASE_NAME}/behavior_out
  * c++_runingtime.txt：仿真器运行时间统计
  * cmp_out*.txt：用于结果对比的文件

#### Code Generator

**编译**

```bash
cd build
make -f code_generator.mak -j
```

**执行**

命令行指令, 例子：

```
./build/code_generator -i test/unit_tests/cases/lenet/lenet.map.txt' -c lenet
```

VSCode执行：

在launch.json中设置输入参数，在Run and Debug活动面板中选择Launch Behavior Simulator并执行。

Executor执行：

```
python flow/executors/execute_code_generator.py
```

**输入**

* -i：必选参数，输入配置文件路径
* -c：必选参数，用例名
* -o：可选参数，输出结果路径名，默认temp/${CASE_NAME}/
* -r：可选参数，是否输出可读的形式，默认为true

**输出**

* temp/${CASE_NAME}/
  * ${CASE_NAME}.code: 生成的Code IR
  * ${CASE_NAME}.code.txt：生成用于调试的Code IR可读版本
