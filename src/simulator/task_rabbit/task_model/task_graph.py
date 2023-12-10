from collections import OrderedDict
from typing import Dict, List, Set, Union

from task_rabbit.task_model.edge import Edge, RearrangeInfo
from task_rabbit.task_model.shape import Shape
from task_rabbit.task_model.task_block import (TaskBlock, TaskBlockType)


class TaskGraph():
    """ 
    任务图，Task IR的具体形式，有任务结点（TaskBlock）和任务之间的边（Edge）组成
    """

    def __init__(self):
        self._nodes: OrderedDict[int, TaskBlock] = OrderedDict()
        # 记录结点分组情况，分组一般用于便捷操作
        self._groups: Dict[int, Set[int]] = {}

        self._inputs = set()
        self._outputs = set()

    @property
    def groups(self):
        return self._groups

    def get_node(self, id: int) -> TaskBlock:
        if id not in self._nodes:
            raise ValueError('Task {:d} is not in the task graph'.format(id))
        if not self._nodes[id].is_enable():
            raise ValueError('Task {:d} is not enable'.format(id))
        return self._nodes.get(id)

    def get_all_node_ids(self):
        return {id for id in self._nodes.keys() if self._nodes[id].is_enable()}

    def get_input_node_ids(self) -> Union[Set[int], int]:
        # 如果输入结点只有一个，则返回该结点ID, 否则返回ID集合
        if len(self._inputs) == 1:
            for id in self._inputs:
                return id
        return self._inputs

    def get_output_node_ids(self) -> Union[Set[int], int]:
        # 如果输出结点只有一个，则返回该结点ID, 否则返回ID集合
        if len(self._outputs) == 1:
            for id in self._outputs:
                return id
        return self._outputs

    def add_node(self, task_node: TaskBlock) -> None:
        """ 
        增加任务结点，增加后，不保证任务图中的结点仍然是拓扑序
        结点之间的连接关系由TaskBlock本身维持，所以本函数不影响边的构建
        Raises:
            ValueError: 当加入的TaskBlock的ID已经存在于TaskGraph时，抛出异常
        """
        if not task_node.is_enable():
            raise ValueError('Task {:d} is not enable'.format(task_node.id))

        id = task_node.id
        if id in self._nodes:
            raise ValueError('Already has node {:d}'.format(id))
        self._nodes[id] = task_node

        if task_node.task_type == TaskBlockType.INPUT:
            self._inputs.add(id)
        if task_node.task_type == TaskBlockType.OUTPUT:
            self._outputs.add(id)

    def connect(self, source_id: int, destination_id: int,
                 source_position: Shape = None, source_size: Shape = None,
                 destination_position: Shape = None, destination_size: Shape = None,
                 rearrange_info: List[RearrangeInfo] = None) -> None:
        """
        增加边连接，即从指定源任务结点的指定边簇（cluster）连接到目的结点的指定边簇
        增加连接，不保证任务图中的结点仍然是拓扑序

        Args:
            source_id: 源任务结点的ID
            destination_id: 目的结点的ID
            source_position: 源结点的边簇中准备通过此连接发送的数据形状起始位置
            source_size: 源结点的边簇中准备通过此连接发送的数据形状大小
            destination_position: 目的结点的边簇中准备接收此连接发送的数据形状起始位置
            destination_size: 目的结点的边簇中准备通过此连接接收的数据形状大小
            rearrange_info: 该条边进行的数据重排操作

        Raises:
            ValueError: 当源结点或目的结点的ID不在TaskGraph中时，抛出异常
            注：两个结点之间可以重复加入多条边，只是在本接口不做限制
        """
        if source_id not in self._nodes:
            raise ValueError('Node ' + source_id + ' not in the graph')
        if destination_id not in self._nodes:
            raise ValueError('Node ' + destination_id + ' not in the graph')

        in_task = self._nodes[source_id]  # type: TaskBlock
        out_task = self._nodes[destination_id]  # type: TaskBlock
        edge = Edge(in_task, out_task, source_position, source_size, \
                    destination_position, destination_size, rearrange_info)

        out_task.add_input_edge(edge)
        in_task.add_output_edge(edge)
        return edge

    def delete_node(self, task_id: int) -> None:
        """
        将task_id表示的结点从Task Graph中移除，即删除所有和该结点相关的边连接
        并将此结点销毁掉（task.destroy()）
        不保证结束后任务图中的结点仍然是拓扑序

        Raises:
            ValueError: 当结点的ID不在TaskGraph中时，抛出异常
        """
        if task_id not in self._nodes:
            raise ValueError('Node ' + task_id + ' not in the graph')

        task = self._nodes.pop(task_id)  # type: TaskBlock

        for in_task in task.all_in_tasks:
            in_task.remove_output_task(task_id)
        for out_task in task.all_out_tasks:
            out_task.remove_input_task(task_id)
        task.destroy()

        if id in self._inputs:
            self._inputs.remove(id)
        if id in self._outputs:
            self._outputs.remove(id)

    def disable_node(self, task_id: int) -> None:
        """
        Disable task_id表示的结点

        Raises:
            ValueError: 当结点的ID不在TaskGraph中时，抛出异常
        """
        if task_id not in self._nodes:
            raise ValueError(task_id + ' not in the graph')

        task = self.get_node(task_id)
        task.disable()

    def enable_node(self, task_id: int) -> None:
        """
        Enable task_id表示的结点

        Raises:
            ValueError: 当结点的ID不在TaskGraph中时，抛出异常
        """
        if task_id not in self._nodes:
            raise ValueError(task_id + ' not in the graph')

        task = self._nodes[task_id]
        task.enable()

    def get_group(self, group_id: int) -> Set[int]:
        if group_id not in self._groups:
            raise ValueError('Group {:d} is not in the task graph'.format(id))
        return self._groups.get(group_id)

    def pop_group(self, group_id: int) -> Set[int]:
        if group_id not in self._groups:
            raise ValueError('Group {:d} is not in the task graph'.format(id))
        return self._groups.pop(group_id)

    def group(self, task_ids: Set[int], group_id: int = None):
        """
        将task_ids包含的任务结点划分到一组
        如果结点已被划分到其他组，将从其他组移除
        如果一个组已经为空，则删除该组
        如果group_id为None，则自动生成一个group_id

        Raises:
            ValueError: 当结点的ID不在TaskGraph中时，抛出异常
        """
        for task_id in task_ids:
            if task_id not in self._nodes:
                raise ValueError('Node ' + task_id + ' not in the graph')
            if not self._nodes[task_id].is_enable():
                raise ValueError('Task {:d} is not enable'.format(task_id))

        empty_groups = set()
        for id, one_group in self._groups.items():
            one_group.difference_update(task_ids)
            if not one_group:
                empty_groups.add(id)

        for id in empty_groups:
            self._groups.pop(id)

        if task_ids:
            new_group_id = 0 if group_id is None else group_id
            if self._groups:
                new_group_id = max(self._groups.keys()) + \
                    1 if group_id is None else group_id
            self._groups[new_group_id] = task_ids

    def topologize(self) -> None:
        """
        将所有结点拓扑排序并重新存到_nodes里

        Raises:
            Exception: 当前不支持带环的图
        """
        in_degrees = {}
        for node_id in self._nodes.keys():
            in_degrees[node_id] = len(self._nodes[node_id].input_edges)
        vertex_num = len(in_degrees)
        node_zero_in = [u for u in in_degrees if in_degrees[u] == 0]
        nodes_topo = OrderedDict()
        while node_zero_in:
            u = node_zero_in.pop()
            nodes_topo[u] = self._nodes[u]
            for edge in nodes_topo[u].output_edges:
                out_task_id = edge.out_task.id
                in_degrees[out_task_id] -= 1
                if in_degrees[out_task_id] == 0:
                    node_zero_in.append(out_task_id)  # 再次筛选入度为0的顶点
        if len(nodes_topo) == vertex_num:  # 如果循环结束后存在非0入度的顶点说明图中有环，不存在拓扑排序
            self._nodes = nodes_topo
        else:
            raise Exception('There\'s a circle in the task graph.')

    def has_connection(self, src_task_id: int, dst_task_id: int) -> bool:
        """
        检查两个任务块之间是否有连接
        Raises:
            ValueError: src_task_id或者dst_task_id不存在于当前的taskgraph中或任务块处于非激活状态
        """
        if src_task_id not in self._nodes:
            raise ValueError(src_task_id + ' not in the graph')
        if dst_task_id not in self._nodes:
            raise ValueError(dst_task_id + ' not in the graph')
        src_task = self.get_node(src_task_id)
        dst_task = self.get_node(dst_task_id)
        if not src_task.is_enable():
            raise ValueError('The source task is not active')
        if not dst_task.is_enable():
            raise ValueError('The destination task is not active')

        return dst_task in src_task.out_tasks

    def get_edge(self, src_task_id: int, dst_task_id: int) -> Edge:
        '''
        评估src_task_id对应的任务块到dst_task_id对应的任务块之间的边
        Raises: 
            ValueError: src_task_id或dst_task_id不在TaskGraph中
            或src_task_id或dst_task_id之间没有边
        '''
        if not self.has_connection(src_task_id, dst_task_id):
            raise ValueError('There is no edge between two tasks')

        source_task = self.get_node(src_task_id)
        for edge in source_task.get_output_edges():
            if edge.out_task.id == dst_task_id and edge.is_enable():
                return edge

    def check(self) -> None:
        """
        检查结点的基本形状信息
        Raises:
            ValueError
        """
        for node in self._nodes.values():
            if node.enable():
                node.check()

    def input(self, tick_num: int) -> Set[TaskBlock]:
        enabled_task = set()
        for input_task in self._inputs:
            input_task.fire(tick_num)
            for next_task in input_task.out_tasks:
                if next_task.activate:
                    enabled_task.add(next_task)
        return enabled_task
            
    def __contains__(self, node_id: str):
        if node_id in self._nodes:
            return True
        return False

    def __iter__(self):
        """
        提供迭代结点的接口
        """
        for task_id, node in self._nodes.items():
            if node.is_enable():
                yield task_id, node

    def __len__(self):
        return len(self._nodes)

    def accept(self, visitor) -> None:
        """
        为访问者模式留下的接口
        """
        for task in self._nodes.values():
            task.accept(visitor)
