import os
import mmap
import sys
sys.path.append("ext/booksim2/api/build")
from greenlet import greenlet
import booksim2
from ext.booksim2.api.booksim_sync import BookSim2Sync

def exp_comp_air(sim: BookSim2Sync, x: float, y: float, iter_num: int):
    x = 2
    y = 2
    iter_num = 5
    sim.inject_comp_air(ca_type=4, data=x, t_inject=1, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=8)
    sim.inject_comp_air(ca_type=4, data=y, t_inject=1, src=1, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=8)
    sim.inject_comp_air(ca_type=4, data=iter_num, t_inject=1, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=1, op_0=8)
    sim.inject_comp_air(ca_type=4, data=1, t_inject=1, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=1, op_0=1+4+8)
    sim.inject_comp_air(ca_type=4, data=1, t_inject=1, src=0, iter_tag=0, pkg_size=1, x_0=1, y_0=1, op_0=8)
    for i in range(5): 
        sim.run_step()
    
    print("----------------")
    # compute
    sim.inject_comp_air(ca_type=0, data=1, t_inject=1, src=0, iter_tag=(iter_num-1), pkg_size=1,   
                                                                x_0=0, y_0=0, op_0=2, 
                                                                x_1=0, y_1=1, op_1=3, 
                                                                x_2=1, y_2=0, op_2=0)
    
    sim.inject_comp_air(ca_type=0, data=1, t_inject=1, src=1, iter_tag=(iter_num-1), pkg_size=1,   
                                                                x_0=0, y_0=0, op_0=2, 
                                                                x_1=-1, y_1=1, op_1=3+4,
                                                                x_2=1, y_2=0, op_2=0)
    
    for i in range(3): 
        sim.run_step()
        print(sim.info)
    sim.run_step(end=True) # End

if __name__ == "__main__":
    sim = BookSim2Sync()
    # manual config (0~3 +-*/ | +8 write reg | +4 go iter)
    
    # e^x: method 1:  78 clocks * (num / bank)
    #             2:  84 / 68 clocks * (2 * num / bank)
    x = 2
    y = 2
    iter_num = 5
    sim.inject_comp_air(ca_type=4, data=x, t_inject=1, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=8)
    sim.inject_comp_air(ca_type=4, data=y, t_inject=1, src=1, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=8)
    sim.inject_comp_air(ca_type=4, data=iter_num, t_inject=1, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=1, op_0=8)
    sim.inject_comp_air(ca_type=4, data=1, t_inject=1, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=1, op_0=1+4+8)
    sim.inject_comp_air(ca_type=4, data=1, t_inject=1, src=0, iter_tag=0, pkg_size=1, x_0=1, y_0=1, op_0=8)
    for i in range(5): 
        sim.run_step()
    
    print("----------------")
    # compute
    sim.inject_comp_air(ca_type=0, data=1, t_inject=1, src=0, iter_tag=(iter_num-1), pkg_size=1,   
                                                                x_0=0, y_0=0, op_0=2, 
                                                                x_1=0, y_1=1, op_1=3, 
                                                                x_2=1, y_2=0, op_2=0)
    
    sim.inject_comp_air(ca_type=0, data=1, t_inject=1, src=1, iter_tag=(iter_num-1), pkg_size=1,   
                                                                x_0=0, y_0=0, op_0=2, 
                                                                x_1=-1, y_1=1, op_1=3+4,
                                                                x_2=1, y_2=0, op_2=0)
    
    for i in range(3): 
        sim.run_step()
        print(sim.info)
    sim.run_step(end=True) # End
    