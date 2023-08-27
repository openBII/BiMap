========================================================================
Getting Started
========================================================================

.. raw:: html

   <style> .python {color:#35abff}
           .cplus {color:#f78864}
           .exe {color:#72adad}
    </style>
.. role:: python
.. role:: cplus
.. role:: exe

工程组织
########################################

Bi-Map分为三个子工程，分别位于三个代码仓库:

* **HNN编程框架：** 编程框架旨在提供一个较通用的、可移植的程序构建方式，没有必要和Bi-Map绑定在一起，所以独立为一个代码仓库。
* **Bi-Map主体工程：** 包括编译栈、仿真栈和测试框架，也包含小规模的示例程序。
* **Bi-Map测试库：** 里面包含各层次测试用例，数据量较大，因此也独立为一个代码仓库。

这三部分工程可以独立构建与使用，也可以根据需求配合使用。每个工程的基本目录结构如下，其中蓝色部分为Python工程，橙色部分为C++工程，青色部分为不开放源码的可执行文件，黑色部分为其他：

- Bi-Map主体工程:
  
  - :cplus:`3rdParties`：第三方库
  - build：工程构建文件
  - docs：文档
  - :python:`flow`：执行器，封装了整个系统的使用接口
  - src：工程主要源码

    - compiler：编译栈

        - ir: 中间表示定义
        - :python:`transformer`: 转换器
        - :python:`mapper`: 映射器
        - :cplus:`code_generator`: 代码生成器
        - :exe:`assembler`: 汇编器
    - runtime：运行时
    - simulator：仿真栈

        - :python:`task_rabbit`: 任务图执行器
        - :python:`resource_simulator`: 资源仿真器
        - :cplus:`behavior_simulator`: 行为级仿真器
        - :exe:`clock_simulator`: 时钟精确级仿真器
        
    - utils：辅助工具

  - temp：代码运行中产生的临时中间文件
  - :python:`test`：测试框架与测试用例生成
  - top：工程的顶层配置，如芯片所包含的核数等
  
- :python:`hybrid_programming_framework`: HNN编程框架
- test_lib: 测试用例库

安装
########################################

HNN编程框架的安装
************************************

**系统要求**

* 64位Linux系统（Ubuntu 20.04或更新）或Windows系统
* Python 3.8+
* `PyTorch <https://pytorch.org/>`_ 1.1或更新
* `惊蛰 <https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.12/>`_ 

.. compound::

  准备就绪后，在shell执行 ::

    git clone TODO 地址
    cd 项目名
    pip install -r requirement.txt

详细使用方式见 :ref:`HNN编程框架`

Bi-Map主体工程的安装
************************************

**系统要求**

* 64位Linux系统（Ubuntu 20.04或更新）
* Python 3.8+
* C++ 2a 或更新

**从源码构建**

构建之前需要安装 `C++的protobuf runtime 和 compiler <https://github.com/protocolbuffers/protobuf/tree/main/src>`_ 及 `python的protobuf runtime <https://github.com/protocolbuffers/protobuf/tree/main/python>`_ 。

.. compound::

  准备就绪后，在shell执行 ::

    git clone TODO 地址
    cd 项目名
    source setenv.sh
    pip install -r requirement.txt
    cd build
    ./compile_all.sh

之后会在当前文件夹中生成一系列可执行程序：

* **code_generator：** 代码生成器可执行文件
* **assembler：** 汇编器可执行文件
* **behavior_simulator：** 行为级仿真器执行文件
* **clock_simulator：** 时钟精确级仿真器执行文件

每一个的可执行文件可直接单独使用，也可以通过封装好的 :ref:`使用接口` 使用。

.. compound::

  如有需要，也可以单独编译一个部分，如单独编译行为级仿真器： ::

    cd 项目名/src/compiler/ir
    ./compile_all_ir.sh    # 先编译所有IR
    cd ../../../build
    make -f behavior_simulator.mak [-j]   #-j选项进行并行编译


顶层配置
########################################

对于Bi-Map主体工程, ``top/config.toml`` 提供了对编译与仿真的顶层配置，包括：

* **路径配置：**
    
  * test_lib: 测试库的根路径
  * temp: 存放临时文件的路径
  * data: 存放数据文件的路径 （详见 :ref:`IO Streamer` ）
  * task_out: Task Rabbit运行任务图的输出结果文件夹
  * map_out: Task Rabbit运行映射图的输出结果文件夹
  * behavior_out: 行为级仿真器输出结果文件夹
  * clock_out: 时钟精确级仿真器输出结果文件夹

* **众核架构配置（对行为层及硬件一致层有效）：**

  * CHIP_X_MAX: 芯片阵列中x方向的芯片个数
  * CHIP_Y_MAX: 芯片阵列中y方向的芯片个数
  * CORE_X_MAX: 芯片中x方向的计算核个数
  * CORE_Y_MAX: 芯片中y方向的计算核个数
  * CORE_MAX: 芯片中总的计算核个数，为CORE_X_MAX * CORE_Y_MAX
  * STEP_GROUP_MAX: 一个芯片可以包含的step group最大个数（详见 :ref:`时空模型` ）
  * PHASE_GROUP_MAX: 一个芯片可以包含的phase group最大个数（详见 :ref:`时空模型` ）
  * PHASE_MAX: 一个step可以包含的phase最大个数 :ref:`时空模型` ）
  * PHASE_ADAPT: bool值，是否将phase group配置为自适应执行模式（详见 :ref:`执行模型` ）

* **存储模块配置（对行为层及硬件一致层有效）：**

  * MEM0_END: Memory 0结束地址 （详见 :ref:`内存模型` ）
  * MEM1_END: Memory 1结束地址
  * MEM2_END: Memory 2结束地址

* **计算模块配置（对行为层及硬件一致层有效）：**

  * MAC_X_SIZE: 线性计算单元x方向并行度 （详见 :ref:`计算模型` ）
  * MAC_Y_SIZE: 线性计算单元y方向并行度

* **路由模块配置：**
  
  * ROUTER_STRATEGY: 路由策略选择 （详见 :ref:`路由模型` ）
  * PACKET_SIZE: 路由包中的数据大小

使用接口
########################################

主要介绍Bi-Map主体工程的使用接口。

组件单独执行
************************************

Bi-Map为所有Python组件和编译好的C++可执行程序提供了Python调用执行接口，我们将调用该接口执行的方式称作通过 ``Executor`` 执行。对于C++ 可执行程序，也可直接调用命令行执行。

**执行转换器**

.. compound::

  Executor执行： ::
    
    python flow/executors/exe_transformer.py

 ``exe_transformer.py`` 中包含两个接口: ``exe_onnx_transform`` 和 ``exe_pytorch_transform`` 。修改 ``exe_transformer.py`` 中的调用函数与调用参数，完成不同用例的执行。

.. py:method:: exe_onnx_transform(onnx_model_path, task_graph_path, [case_name=None, optimize_config=None, readable_result=True])

  调用转换器，将ONNX模型转换为任务图模型（Task IR）。

  :param str onnx_model_path: 需要被转换的ONNX模型的完成路径（文件路径+文件名）
  :param str task_graph_path: 转换后的任务图的输出完成路径。
  :param str case_name: 此次运行的用例名，如果为None，则采用ONNX的文件名作为用例名
  :param OptimizeConfig optimize_config: 转换过程中的优化选项，包括
    
    * bool merge_relu_maxpool: 是否融合RuLU-Max Pooling为一个任务图结点
    * bool optimize_conv_storage: 是否优化卷积的内存排布
  
  :param bool readable_result: 是否输出人类可读的任务图文件

  :return: 返回转换是否成功。转换后的结果直接生成相应的文件，如果 ``readable_result`` 为 ``False``, 则只生成 ``{case_name}.task`` 任务图文件; 如果 ``readable_result`` 为 ``True``, 则还会生成人类可读的 ``{case_name}.task.txt`` 文件
  
  :使用示例: 

    .. code:: python

      onnx_model_path = '$test_lib/model_lib/Lenet.onnx'
      task_graph_path = './temp/lenet.task'
      optimizer = OptimizeConfig()
      optimizer.merge_relu_maxpool = True
      optimizer.optimize_conv_storage = True

      exe_onnx_transform(onnx_model_path, task_graph_path, optimize_config = optimizer)

    上述得到的 `lenet.task` 的图形化为如下：

    .. image::  _static/case1.png
     :width: 100%
     :align: center

.. py:method:: exe_pytorch_transform(pytorch_model, task_graph_path, input, [case_name=None, pretrained_model_path=None, reserve_control_flow=False, optimize_config=None, readable_result=True])

  调用转换器，将PyTorch模型转换为任务图模型（Task IR）。

  :param pytorch_model: 需要被转换的PyTorch模型
  :param str task_graph_path: 转换后的任务图的输出完成路径
  :param input: PyTorch模型的输入数据
  :param str case_name: 此次运行的用例名，如果为None，则采用任务图的文件名作为用例名
  :param str pretrained_model_path: 预训练模型的完成路径，如果是已量化模型，预训练模型可以包含量化参数
  :param bool reserve_control_flow: 是否使用 ``torch.jit.script`` 包含PyTorch模型中的控制流
  :param OptimizeConfig optimize_config: 转换过程中的优化选项，与 ``exe_onnx_transform`` 中的用法一致。
  :param bool readable_result: 是否输出人类可读的任务图文件
  :type pytorch_model: torch.nn.Module or QModel or SQModel
  :type input: torch.Tensor or Tuple[torch.Tensor]
  :return: 返回转换是否成功。转换后的结果直接生成相应的文件，如果 ``readable_result`` 为 ``False``, 则只生成 ``*.task`` 任务图文件; 如果 ``readable_result`` 为 ``True``, 则还会生成人类可读的 ``*.task.txt`` 文件
  
  :使用示例: 

    .. code:: python

      pytorch_model = Resnet50()  # PyTorch NN model
      x = Tensor...
      task_graph_path = './temp/resnet50.task'
      optimizer = OptimizeConfig()
      optimizer.merge_relu_maxpool = True
      optimizer.optimize_conv_storage = True

      exe_pytorch_transform(pytorch_model, task_graph_path, x, optimize_config = optimizer)

**执行Task Rabbit**

.. compound::

  Executor执行： ::
    
    python flow/executors/exe_task_rabbit.py

``exe_task_rabbit.py`` 中包含两个接口: ``exe_task_rabbit_with_task`` 和 ``exe_task_rabbit_with_map`` 。

.. py:method:: exe_task_rabbit_with_task(task_graph_path, case_name[, input_path=None, output_dir=None])

  调用Task Rabbit，根据输入数据前向推理任务图，得到推理结果。

  :param str task_graph_path: 需要前向推理的任务图的完成路径（文件路径+文件名）
  :param str case_name: 此次运行的用例名
  :param str input_path: 任务图输入数据的完整路径，如果为None，则任务图的起始输入数据块应该有预存的数据，Task Rabbit按照此数据进行前向推理。
  :param str output_dir: 输出结果存放文件夹，如果为None，则默认在工程目录下创建 ``temp/{case_name}/task_out`` 文件夹，并将结果保存在这里。
  :return: 返回前向推理是否成功。在指定或默认的 ``output_dir`` 中输出前向推理结果。任务图可能包含多个较为独立的网络模型，每个算法模型可以有多个输出（对应多输出网络模型），并可能连续执行多个输入样本。我们用 ``net_id`` 、 ``socket_id``、 ``frame_id`` 分别表示网络、输出端口、输入样本。相应输出结果的文件以 ``o_{net_id}_{socket_id}_{frame_id}.dat`` 的形式命名。
  
  :使用示例: 

    .. code:: python

      task_graph_path = './temp/resnet50.task'
      case_name = 'resnet50'

      exe_task_rabbit_with_task(task_graph_path, case_name)

.. py:method:: exe_task_rabbit_with_map(map_graph_path, case_name[, input_path=None, output_dir=None])

  调用Task Rabbit，根据输入数据前向推理映射图中包含的任务图，得到推理结果。映射图由两部分组成，一部分是任务图，另一部分是任务图在硬件时空资源的表示上的映射。第二部分不影响数据结果，所以Task Rabbit可以输入映射图，得到前向推理结果。

  :param str map_graph_path: 需要前向推理的映射图的完成路径（文件路径+文件名）
  :param str case_name: 此次运行的用例名
  :param str input_path: 映射图图输入数据的完整路径，如果为None，则映射图起始输入数据块应该有预存的数据，Task Rabbit按照此数据进行前向推理。
  :param str output_dir: 输出结果存放文件夹，如果为None，则默认在工程目录下创建 ``temp/{case_name}/task_out`` 文件夹，并将结果保存在这里。
  :return: 返回前向推理是否成功。输出结果命名方式同 ``exe_task_rabbit_with_task``
  
  :使用示例: 

    .. code:: python

      map_graph_path = './temp/resnet50.map'
      case_name = 'resnet50'

      exe_task_rabbit_with_map(map_graph_path, case_name)



**映射器之后再加**


**执行代码生成器**

.. compound::

  Executor执行： ::
    
    python flow/executors/exe_code_generator.py

.. py:method:: exe_code_generator(map_graph_path, case_name[, output_dir=None, readable_result=True])

  调用代码生成器，将Mapping IR（映射图）编译为Assembly IR。

  :param str map_graph_path: 需要前向推理的映射图的完成路径（文件路径+文件名）
  :param str case_name: 此次运行的用例名
  :param str output_dir: 编译结果存放文件夹，如果为None，则默认在工程目录下创建 ``temp/{case_name}`` 文件夹，并将结果保存在这里。
  :param bool readable_result: 是否输出人类可读的Assembly IR文件
  :return: 返回代码生成是否成功。代码生成器的结果直接生成相应的文件，如果 ``readable_result`` 为 ``False``, 则只生成 ``{case_name}.asm`` 任务图文件; 如果 ``readable_result`` 为 ``True``, 则还会生成人类可读的 ``{case_name}.asm.txt`` 文件
  
  :使用示例: 

    .. code:: python

      map_graph_path = './temp/resnet50.map'
      case_name = 'resnet50'

      exe_code_generator(map_graph_path, case_name)

.. compound::

  命令行执行： ::

    build/code_generator -i "./temp/resnet50.map" -c "resnet50"
    
  可配置输入参数（含义同相应的Executor接口参数）：

  * -i 同 ``map_graph_path`` 必须
  * -c 同 ``case_name`` 必须
  * -o 同 ``output_dir`` 可选
  * -r 同 ``readable_result`` 可选

**执行行为级仿真器**

.. compound::

  Executor执行： ::
    
    python flow/executors/exe_behavior_simulator.py

.. py:method:: exe_behavior_simulator(assembly_ir_path, case_name[, output_dir=None])

  调用行为级仿真器，输入Assembly IR进行仿真，输出仿真执行的结果。

  :param str assembly_ir_path: 需要进行仿真的Assembly IR的完成路径（文件路径+文件名）
  :param str case_name: 此次运行的用例名
  :param str output_dir: 仿真结果存放文件夹，如果为None，则默认在工程目录下创建 ``temp/{case_name}/behavior_out`` 文件夹，并将结果保存在这里。
  :return: 返回仿真是否成功。在指定或默认的 ``output_dir`` 中输出仿真结果。Assembly IR中会指定改用例是 ``CASE_OUT`` 模式还是 ``PRIM_OUT`` 模式。 ``CASE_OUT`` 模式下的输出格式同 Task Rabbit，``PRIM_OUT`` 模式下的输出格式会在 :ref:`行为级仿真器` 中详细介绍。
  
  :使用示例: 

    .. code:: python

      map_graph_path = './temp/resnet50.asm'
      case_name = 'resnet50'

      exe_behavior_simulator(assembly_ir_path, case_name)

.. compound::

  命令行执行： ::

    build/behavior_simulator -i "./temp/resnet50.asm" -c "resnet50"
    
  可配置输入参数（含义同相应的Executor接口参数）：

  * -i 同 ``assembly_ir_path`` 必须
  * -c 同 ``case_name`` 必须
  * -o 同 ``output_dir`` 可选
  * -d 选择处理器加速仿真执行，默认为 "cpu" , 表示全部在cpu上执行仿真器。 "gpu" 表示使用GPU加速某些过程，目前还没有实现。

**执行汇编器**

.. compound::

  Executor执行： ::
    
    python flow/executors/exe_assembler.py

.. py:method:: exe_assembler(assembly_ir_path, case_name[, output_dir=None, readable_result=True])

  调用汇编器，将Assembly IR汇编为硬件可执行的Code IR。

  :param str assembly_ir_path: 需要进行汇编的Assembly IR的完成路径（文件路径+文件名）
  :param str case_name: 此次运行的用例名
  :param str output_dir: 汇编结果存放文件夹，如果为None，则默认在工程目录下创建 ``temp/{case_name}`` 文件夹，并将结果保存在这里。
  :param bool readable_result: 是否输出人类可读的Code IR文件
  :return: 返回汇编是否成功。汇编器的结果直接生成相应的文件，如果 ``readable_result`` 为 ``False``, 则只生成 ``{case_name}.code`` 任务图文件; 如果 ``readable_result`` 为 ``True``, 则还会生成人类可读的 ``{case_name}.code.txt`` 文件。
  
  :使用示例: 

    .. code:: python

      map_graph_path = './temp/resnet50.asm'
      case_name = 'resnet50'

      exe_assembler(map_graph_path, case_name)

.. compound::

  命令行执行： ::

    build/assembler -i "./temp/resnet50.asm" -c "resnet50"
    
  可配置输入参数（含义同相应的Executor接口参数）：

  * -i 同 ``assembly_ir_path`` 必须
  * -c 同 ``case_name`` 必须
  * -o 同 ``output_dir`` 可选
  * -r 同 ``readable_result`` 可选

**执行时钟精确级仿真器**

.. compound::

  Executor执行： ::
    
    python flow/executors/exe_clock_simulator.py

.. py:method:: exe_clock_simulator(code_ir_path, case_name[, output_dir=None])

  调用时钟精确级仿真器，输入Code IR进行仿真，输出仿真执行的结果和一些调试的信息。

  :param str code_ir_path: 需要进行仿真的Code IR的完成路径（文件路径+文件名）
  :param str case_name: 此次运行的用例名
  :param str output_dir: 仿真结果存放文件夹，如果为None，则默认在工程目录下创建 ``temp/{case_name}/clock_out`` 文件夹，并将结果保存在这里。在 ``temp/{case_name}`` 文件夹下输出一些调试结果。
  :return: 返回仿真是否成功。在指定或默认的 ``output_dir`` 中输出仿真结果。Code IR中会指定改用例是 ``CASE_OUT`` 模式还是 ``PRIM_OUT`` 模式。 ``CASE_OUT`` 模式下的输出格式同 Task Rabbit，``PRIM_OUT`` 模式下的输出格式和行为级仿真器一致，会在 :ref:`行为级仿真器` 中详细介绍。除此之外，始终精确级仿真器生成的其他文件 会在 :ref:`时钟精确级仿真器` 中介绍。
  
  :使用示例: 

    .. code:: python

      map_graph_path = './temp/resnet50.code'
      case_name = 'resnet50'

      exe_clock_simulator(code_ir_path, case_name)

.. compound::

  命令行执行 ::

    build/clock_simulator -i "./temp/resnet50.asm" -c "resnet50"
    
  可配置输入参数（含义同相应的Executor接口参数）：

  * -i 同 ``code_ir_path`` 必须
  * -c 同 ``case_name`` 必须
  * -o 同 ``output_dir`` 可选

组件串联执行
************************************

在实际使用过程中，往往需要几个组件联系来使用，如 ``转换器-映射器-代码生成器-汇编器`` 完成完整的编译过程。这样的一个多组件串联执行过程我们称作一个执行流。 ``flow/executors`` 中提供了常见执行流的执行接口， 如

.. compound::

  执行完整的编译过程： ::
    
    python flow/executors/exe_compiler.py

.. py:method:: exe_compiler(onnx_model_path, case_name[, output_dir=None])

  :param str onnx_model_path: 需要被编译的ONNX模型的完成路径（文件路径+文件名）
  :param str case_name: 此次运行的用例名
  :param str output_dir: 编译结果存放文件夹，如果为None，则默认在工程目录下创建 ``temp/{case_name}`` 文件夹，并将编译结果保存在这里。
  :param CompileOption compile_option: 编译选项, TODO。

.. image::  _static/framework2.png
   :width: 100%
   :align: center

``flow/execute.py`` 中提供了任意执行流的执行接口。上图中展示了每个组件对应的编号，如代码生成器的编号为 ``C3``。我们用这些编号组成的字符串表示一个执行流，如 ``C3-S4-C4-S5`` 表示执行 ``代码生成器-行为级仿真器-汇编器-代码生成器``。 该功能通过如下接口实现：

.. py:class:: Execution

  .. py:method:: auto_execute(flow_id, case_path, case_name, *args)

    自动执行一个执行流。

    :param str flow_id: 组件编号组成的执行流字符串表示
    :param str case_path:  执行流中第一个组件对应的输入文件的完整路径（文件路径+文件名）
    :param str case_name: 此次运行的用例名
    :param args: 执行流中第一个组件对应的 ``Executor`` 接口函数的除了 `case_path` 与 `case_name` 之外的其它输入参数

    :return: 返回是否所有组件执行成功。每个组件的结果放在相应的文件夹里。

    Execution类维护三个字典：

    * FLOW_MAP：将每个组件的编号对应到这个组件的 ``Executor`` 执行函数上。
    * INPUT_MAP：在自动执行过程中，每个组件的默认输入参数，包括输入文件路径等。
    * OUTPUT_MAP：在自动执行过程中，每个组件的默认输出位置与输出文件名。

    在预设的情况下，在一个合法的执行流中，后执行的组件所需的输入文件一定是前面的组件在相同位置的输出文件。如有更灵活的使用需求，可以修改 ``INPUT_MAP`` 与 ``OUTPUT_MAP`` 的内容。

    :使用示例: 

      .. code:: python

        flow_id = 'C3-S4-C4-S5'
        case_path = './temp/resnet50.map'
        case_name = 'resnet50'
        Execution.auto_execute(flow_id, case_path, case_name)
