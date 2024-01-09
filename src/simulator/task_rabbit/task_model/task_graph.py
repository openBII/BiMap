from collections import OrderedDict
from typing import Dict, List, Set, Union

from src.simulator.task_rabbit.task_model.edge import Edge, RearrangeInfo
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.task_block import (TaskBlock, TaskBlockType)
from src.simulator.task_rabbit.task_model.input_type import InputType


class TaskGraph():
    """An implementation of Task IR, which is composed of TaskBlocks and Edges.
    """

    def __init__(self):
        self._nodes: OrderedDict[int, TaskBlock] = OrderedDict()
        # groups of nodes for convenience
        self._groups: Dict[int, Set[int]] = {}

        self._inputs: Set[int] = set()
        self._outputs: Set[int] = set()

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
        """Return the IDs of input nodes.

        If there is only one input node, this method will return the ID of this node. 
        Otherwise, it will return the set of IDs of the input nodes.
        """
        if len(self._inputs) == 1:
            for id in self._inputs:
                return id
        return self._inputs

    def get_output_node_ids(self) -> Union[Set[int], int]:
        """Return the IDs of output nodes.

        If there is only one output node, this method will return the ID of this node. 
        Otherwise, it will return the set of IDs of the output nodes.
        """
        if len(self._outputs) == 1:
            for id in self._outputs:
                return id
        return self._outputs

    def add_node(self, task_node: TaskBlock) -> None:
        """Add node into the task graph.

        After adding a node, the set of nodes is not in topological order.
        The connections between nodes are maintained by the TaskBlock itself, 
        so this method will not influence the construction of edges.

        Raises:
            ValueError: raise when the node already exists or the node is not enabled.
        """
        if not task_node.is_enable():
            raise ValueError('Task {:d} is not enabled'.format(task_node.id))

        id = task_node.id
        if id in self._nodes:
            raise ValueError('Task {:d} already in this task graph'.format(id))
        self._nodes[id] = task_node

        if task_node.task_type == TaskBlockType.INPUT:
            self._inputs.add(id)
        if task_node.task_type == TaskBlockType.OUTPUT:
            self._outputs.add(id)

    def connect(self, source_id: int, destination_id: int,
                 source_position: Shape = None, source_size: Shape = None,
                 destination_position: Shape = None, destination_size: Shape = None,
                 rearrange_info: List[RearrangeInfo] = None) -> Edge:
        """Add a edge between Task source_id and Task destination_id

        The edge will connect a slice of the output data of Task source_id to 
        a slice of the input data of Task destination_id.

        Notice: Two nodes can be connected by multiple edges.

        Args:
            source_id: ID of the source node.
            destination_id: ID of the sink node.
            source_position: The starting position of the slice of the output data of the source node.
            source_size: The size of the slice of the output data of the source node.
            destination_position: The starting position of the slice of the input data of the sink node.
            destination_size: The size of the slice of the input data of the sink node.
            rearrange_info: Information on data rearrangement operations in this edge.

        Raises:
            ValueError: raise when the source node or the sink node is not in the task graph.
        """
        if source_id not in self._nodes:
            raise ValueError('Task ' + str(source_id) + ' not in the graph')
        if destination_id not in self._nodes:
            raise ValueError('Task ' + str(destination_id) + ' not in the graph')

        in_task = self._nodes[source_id]
        out_task = self._nodes[destination_id]
        edge = Edge(in_task, out_task, source_position, source_size, \
                    destination_position, destination_size, rearrange_info)

        out_task.add_input_edge(edge)
        in_task.add_output_edge(edge)
        return edge

    # FIXME: 这个方法有问题，all_in_tasks这个方法被注释掉了
    def delete_node(self, task_id: int) -> None:
        """
        将task_id表示的结点从Task Graph中移除，即删除所有和该结点相关的边连接
        并将此结点销毁掉（task.destroy()）
        不保证结束后任务图中的结点仍然是拓扑序

        Raises:
            ValueError: 当结点的ID不在TaskGraph中时，抛出异常
        """
        if task_id not in self._nodes:
            raise ValueError('Task ' + str(task_id) + ' not in the graph')

        task = self._nodes.pop(task_id) 

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
        """Disable the node whose ID is task_id.

        Raises:
            ValueError: raise when the input ID is not in the task graph.
        """
        if task_id not in self._nodes:
            raise ValueError('Task ' + str(task_id) + ' not in the graph')

        task = self.get_node(task_id)
        task.disable()

    def enable_node(self, task_id: int) -> None:
        """Enable the node whose ID is task_id.

        Raises:
            ValueError: raise when the input ID is not in the task graph.
        """
        if task_id not in self._nodes:
            raise ValueError('Task ' + str(task_id) + ' not in the graph')

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
                raise ValueError('Task ' + str(task_id) + ' not in the graph')
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

    # TODO: implement this in C++
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
        for edge in source_task.output_edges:
            if edge.out_task.id == dst_task_id and edge.is_enable():
                return edge

    # FIXME TaskBlock的check方法未定义
    def check(self) -> None:
        """
        检查结点的基本形状信息
        Raises:
            ValueError
        """
        for node in self._nodes.values():
            if node.enable():
                node.check()

    def input(self, tick_num: int, input_type: InputType) -> Set[TaskBlock]:
        """The task graph will accept tick_num ticks.

        Args:
            tick_num: int, number of input ticks.

        Returns:
            activated_tasks: Set[TaskBlock], which tasks will be activated after injecting the input ticks.
        """
        activated_tasks = set()
        for input_task_id in self._inputs:
            input_task = self._nodes[input_task_id]
            input_task.fire(tick_num, input_type)
            for next_task in input_task.out_tasks:
                if next_task.activated:
                    activated_tasks.add(next_task)
        return activated_tasks
    
    def get_activated_tasks(self, fired_tasks: Set[TaskBlock]) -> Set[TaskBlock]:
        """Returns the activated tasks after the firing of given tasks.

        Args:
            fired_tasks: Set[TaskBlock], input fired tasks.

        Returns:
            activated_tasks: Set[TaskBlock], tasks activated after the firing of input tasks.
        """
        activated_tasks = set()
        for fired_task in fired_tasks:
            for next_task in fired_task.out_tasks:
                if next_task.activated:
                    activated_tasks.add(next_task)
        return activated_tasks
            
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
