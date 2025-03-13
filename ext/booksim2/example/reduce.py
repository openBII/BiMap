import os
import mmap
import sys
sys.path.append("ext/booksim2/api/build")
from greenlet import greenlet
import booksim2
from ext.booksim2.api.booksim_sync import BookSim2Sync

if __name__ == "__main__":
    sim = BookSim2Sync()
    # manual config (0~3 +-*/ | +8 write reg | +4 go iter)
    # L0 (x, y)
    # - (0,0) (0,2) -> (0,0) # 2
    # - (2,0) (2,2) -> (2,0) # 2
    # L1 (x, y)
    # - (0,0) (2,0) -> (2,2) # 2
    
    sim.inject_comp_air(ca_type=4, data=2, t_inject=1, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=1+4+8)
    sim.inject_comp_air(ca_type=4, data=2, t_inject=1, src=2, iter_tag=0, pkg_size=1, x_0=0, y_0=2, op_0=1+4+8)
    sim.inject_comp_air(ca_type=4, data=2, t_inject=1, src=2, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=1+4+8)
    for i in range(4): 
        sim.run_step()
    
    print("----------------")
    # reduce
    # (0,0) -> 1
    sim.inject_comp_air(ca_type=1, data=1, t_inject=1, src=0, iter_tag=0, pkg_size=1,   
                                                                x_0=0, y_0=0, op_0=0+8, 
                                                                x_1=2, y_1=2, op_1=0+8) # 3
    # (0,2) -> 2
    sim.inject_comp_air(ca_type=1, data=2, t_inject=1, src=16, iter_tag=0, pkg_size=1,   
                                                                x_0=0, y_0=-2, op_0=0+8, 
                                                                x_1=2, y_1=2, op_1=0+8) # 4
    
    # (2,0) -> 3
    sim.inject_comp_air(ca_type=1, data=3, t_inject=1, src=2, iter_tag=0, pkg_size=1,   
                                                                x_0=0, y_0=0, op_0=0+8, 
                                                                x_1=0, y_1=2, op_1=0+8) # 5
    # (2,2) -> 4
    sim.inject_comp_air(ca_type=1, data=4, t_inject=1, src=18, iter_tag=0, pkg_size=1,   
                                                                x_0=0, y_0=-2, op_0=0+8, 
                                                                x_1=0, y_1=2, op_1=0+8) # 6
    
    for i in range(2): 
        sim.run_step()
        print(sim.info)
    
    # End
    sim.run_step(end=True)
    