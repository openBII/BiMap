#!/usr/bin/env python
# coding: utf-8

"""
SpaceColumn类是一种自定义的list，表示一个空间列
具体的说，在tianjicX架构中，表示一个Core model
"""


class MemoryRule():
    PRIMITIVE_SIZE = 32 * 4 * 32     # Byte
    SIZE = 144 * 1024 - PRIMITIVE_SIZE

    MEM0 = 64 * 1024
    MEM1 = 64 * 1024
    MEM2 = 16 * 1024 - PRIMITIVE_SIZE
