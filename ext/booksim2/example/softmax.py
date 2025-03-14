import os
import mmap
import sys
sys.path.append("ext/booksim2/api/build")
from greenlet import greenlet
import booksim2
from ext.booksim2.api.booksim_sync import BookSim2Sync
from ext.booksim2.example.broadcast import broadcast_comp_air
from ext.booksim2.example.reduce import reduce_comp_air
from ext.booksim2.example.exp import exp_comp_air
from ext.booksim2.example.scalar import scalar_r_comp_air

def softmax_comp_air(sim: BookSim2Sync):
    # e^s 8 num/bank * 16 banks * 32 devices
    #     2 num/round
    exp_r0 = exp_comp_air(sim, 2, 2, 4, True)
    exp_r1 = exp_comp_air(sim, 2, 2, 4, False)
    exp_r2 = exp_comp_air(sim, 2, 2, 4, False)
    exp_r3 = exp_comp_air(sim, 2, 2, 4, False)
    
    # reduce add
    reduce = reduce_comp_air(sim, sources=16, step=2, data=2.5, para_num=4)
    
    # broadcast
    broadcast_comp_air(sim, src=0, targs=4, step=1, data=2.5)
    broadcast_comp_air(sim, src=0, targs=8, step=4, data=2.5)
    broadcast_comp_air(sim, src=0, targs=2, step=2, data=2.5)
    
    # 1 / reduce (Set it here behind broadcasts for showing the time)
    div_r = scalar_r_comp_air(sim, 2.3, 3)
    
    # Tim
    print("[exp_r0]", exp_r0)
    print("[exp_r1]", exp_r1)
    print("[exp_r2]", exp_r2)
    print("[exp_r3]", exp_r3)
    print("[reduce]", reduce)
    print("[div_r]", div_r)

if __name__ == "__main__":
    sim = BookSim2Sync()
    softmax_comp_air(sim)