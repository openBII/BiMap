import os
import mmap
import sys
sys.path.append("ext/booksim2/api/build")
from greenlet import greenlet
import booksim2
from ext.booksim2.api.booksim_sync import BookSim2Sync

if __name__ == "__main__":
    sim = BookSim2Sync()
    # Broadcast (0-7) 34 clocks
    # Goal:
    # 0 (src)
    # 0       8
    # 0   4   8    12
    # 0 2 4 6 8 10 12 14
    sim.inject_comp_air(ca_type=3, data=1.5, t_inject=1, src=0, iter_tag=0, pkg_size=1,
                                                                x_0=0, y_0=0, op_0=0+8,
                                                                x_1=2, y_1=0, op_1=0+8,
                                                                x_2=4, y_2=0, op_2=0+8,
                                                                x_3=8, y_3=0, op_3=0+8)
    for i in range(5): 
        sim.run_step()
        print(sim.info)
    sim.run_step(end=True) # End