import os
import math
import sys
sys.path.append("ext/booksim2/api/build")
from greenlet import greenlet
import booksim2
from ext.booksim2.api.booksim_sync import BookSim2Sync
from ext.booksim2.example.broadcast import broadcast_comp_air
from ext.booksim2.example.reduce import reduce_comp_air
from ext.booksim2.example.sqrt import sqrt_comp_air
from ext.booksim2.example.scalar import scalar_r_comp_air


def rms_norm_comp_air(sim: BookSim2Sync):
    """
    Perform Root Mean Square (RMS) normalization computation using BookSim2 simulator.

    Args:
        sim (BookSim2Sync): An instance of the BookSim2Sync simulator.

    This function calculates the RMS normalization using a series of computational steps
    including reduction, square root calculation, broadcasting, and scalar multiplication.
    """
    
    #  1 / sqrt(sum(x^2) / n)
    #  1 / (sqrt(x^2) * k); k = 1 / sqrt(n)
    
    # reduce add
    reduce = reduce_comp_air(sim, sources=16, step=2, data=2.5, para_num=4)
    
    # sqrt
    iter_num = 4
    srqt0 = sqrt_comp_air(sim, 2.5, iter_num, True)
    srqt1 = sqrt_comp_air(sim, 2.5, iter_num, False)
    srqt2 = sqrt_comp_air(sim, 2.5, iter_num, False)
    srqt3 = sqrt_comp_air(sim, 2.5, iter_num, False)
    
    # broadcast
    broadcast_comp_air(sim, src=0, targs=4, step=1, data=2.5)
    broadcast_comp_air(sim, src=0, targs=8, step=4, data=2.5)
    broadcast_comp_air(sim, src=0, targs=2, step=2, data=2.5)
    
    # * 1/sqrt(N) (Set it here behind broadcasts for showing the time)
    mul = scalar_r_comp_air(sim, 1/math.sqrt(4096), 2)
    
    # Tim
    print("[reduce]", reduce)
    print("[srqt0]", srqt0)
    print("[srqt1]", srqt1)
    print("[srqt2]", srqt2)
    print("[srqt3]", srqt3)
    print("[mul]", mul)
    
if __name__ == "__main__":
    sim = BookSim2Sync()
    rms_norm_comp_air(sim)