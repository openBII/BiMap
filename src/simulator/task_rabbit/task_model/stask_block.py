from src.simulator.task_rabbit.task_model.task_block_type import TaskBlockType
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.storage import Storage
from src.simulator.task_rabbit.task_model.task_block import TaskBlock


class STaskBlock(TaskBlock):
    """
    STaskBlock类描述一个存储任务块
    """

    def __init__(self, task_id: int, shape: Shape, precision: Precision):
        super().__init__(task_id, shape, precision)
        self._pipeline_area = shape 
        self._type = TaskBlockType.SI

    @property
    def pipeline_num(self) -> Shape:
        return self._pipeline_num

    @pipeline_num.setter
    def pipeline_num(self, value: Shape):
        '''
        设置流水区域，流水更新后，需要重新构建存储信息
        raises:
            ValueError: 当设置的行流水参数大于任务块y方向大小时，抛出异常
        '''
        self._pipeline_num = value
        self._construct_storage()

    def _construct_computation(self) -> None:
        self._computation = 0

    def _construct_storage(self) -> None:
        # 加入计算过程
        self._storage = 0

    def accept(self, visitor):
        visitor.visit_S(self)

    # def copy_like(self) -> TaskBlock:
    #     data = deepcopy(self._data)
    #     new_task_block = SITaskBlock(IDGenerator.get_next_task_id(),
    #                                  copy(self.shape),
    #                                  self.precision, data)
    #     return new_task_block