import operator
from functools import reduce
from typing import Tuple


class Shape():
    """
    形状，Task模型中基本的数据结构
    可以包含y、x、f、r、ky、kx 6个独立维度的大小
    可以冗余的记录iy与ix维度的大小
    如果某个维度大小为0，则意味着不存在这一维度
    """

    def __init__(self, nx=0, nr=0, nf=0, ny=0, nky=0, nkx=0, niy=0, nix=0, batch=1):
        self.nx = nx
        self.nr = nr
        self.nf = nf
        self.ny = ny
        self.nky = nky
        self.nkx = nkx

        self.niy = niy
        self.nix = nix

        self.batch = batch

        # self.additional_dims    # for extension

        # 不能出现小于0的维度
        assert not [dim for dim in self.dim_tuple if dim < 0]


    # Transformer specific
    @property
    def ns(self) -> int:
        return self.nx
    
    @property
    def nh(self) -> int:
        return self.nr
    
    @property
    def na(self) -> int:
        return self.nf

    @ns.setter
    def ns(self, value):
        self.nx = value

    @nh.setter
    def nh(self, value):
        self.nh = value
        
    @na.setter
    def na(self, value):
        self.na = value

    @property
    def dim_tuple(self) -> Tuple[int]:
        return (self.ny, self.nx, self.nf, self.nr, self.nky, self.nkx)

    @property
    def dim_num(self) -> int:
        # 有效维度的个数
        return len([dim for dim in self.dim_tuple if dim > 0])

    @property
    def window_size(self) -> int:
        return self.nr * self.nkx * self.nky

    @property
    def volume(self) -> int:
        positive_dims = [dim for dim in self.dim_tuple if dim > 0]
        if not positive_dims:
            return 0  # 所有维度都为0的Shape
        return reduce(lambda a, b: a * b, positive_dims)

    @staticmethod
    def max(s1, s2):
        return Shape(*map(max, zip(s1.dim_tuple, s2.dim_tuple)))

    @staticmethod
    def min(s1, s2):
        return Shape(*map(min, zip(s1.dim_tuple, s2.dim_tuple)))

    def __str__(self) -> str:
        return str(self.dim_tuple)

    def __getitem__(self, idx) -> int:
        if idx > 7 or idx < -8:
            raise IndexError('Index {:d} is out of range'.format(idx))
        idx = idx + 8 if idx < 0 else idx
        return (self.dim_tuple + (self.niy, self.nix))[idx]

    def __setitem__(self, idx, value) -> None:
        if idx > 7 or idx < -8:
            raise IndexError('Index {:d} is out of range'.format(idx))
        idx = idx + 8 if idx < 0 else idx
        if idx == 0:
            self.ny = value
        elif idx == 1:
            self.nx = value
        elif idx == 2:
            self.nf = value
        elif idx == 3:
            self.nr = value
        elif idx == 4:
            self.nky = value
        elif idx == 5:
            self.nkx = value
        elif idx == 6:
            self.niy = value
        elif idx == 7:
            self.nix = value

    def __eq__(self, other):
        return (all(x == y for x, y in zip(self.dim_tuple, other.dim_tuple)) and
                self.niy == other.niy and self.nix == other.nix)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        '''
        x + y == 0 希望判断x == 0, y == 0 (因为x、y都不能为负数)
        x == 0, y == 0意味着当前维度可能不存在，不应该算入比较之中
        '''
        return all(x > y or x + y == 0 for x, y in zip(self.dim_tuple, other.dim_tuple))

    def __ge__(self, other):
        return all(x >= y for x, y in zip(self.dim_tuple, other.dim_tuple))

    def __lt__(self, other):
        return all(x < y or x + y == 0 for x, y in zip(self.dim_tuple, other.dim_tuple))

    def __le__(self, other):
        return all(x <= y for x, y in zip(self.dim_tuple, other.dim_tuple))

    def __add__(self, other):
        if isinstance(other, int):
            return self.__radd__(other)
        return Shape(*map(sum, zip(self.dim_tuple, other.dim_tuple)))

    def __sub__(self, other):
        if isinstance(other, int):
            return self.__rsub__(other)
        return Shape(*map(operator.sub, self.dim_tuple, other.dim_tuple))

    # 除以0没有解决
    # def __floordiv__(self, other):
    #     if not isinstance(other, Shape):
    #         other = Shape(other, other, other, other, other, other)
    #     return Shape(self.ny // other.ny, self.nx // other.nx,
    #                  self.nf // other.nf, self.nr // other.nr,
    #                  self.nky // other.nky, self.nkx // other.nkx)

    # def __mod__(self, other):
    #     if not isinstance(other, Shape):
    #         other = Shape(other, other, other, other, other, other)
    #     return Shape(self.ny % other.ny, self.nx % other.nx,
    #                  self.nf % other.nf, self.nr % other.nr,
    #                  self.nky % other.nky, self.nkx % other.nkx)

    def __radd__(self, other: int):
        return Shape(*(x + other for x in self.dim_tuple))

    def __rsub__(self, other: int):
        return Shape(*(x - other for x in self.dim_tuple))

    # def __rmul__(self, other: int):
    #     return Shape(self.ny * other, self.nx * other, self.nf * other,
    #                  self.nr * other, self.nky * other, self.nkx * other)

    # def __rfloordiv__(self, other):
    #     return Shape(self.ny // other, self.nx // other, self.nf // other,
    #                  self.nr // other, self.nky // other, self.nkx // other)

    # def __rmod__(self, other):
    #     return Shape(self.ny % other, self.nx % other, self.nf % other,
    #                  self.nr % other, self.nky % other, self.nkx % other)
