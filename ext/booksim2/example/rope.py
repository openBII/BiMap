import os
import mmap
import sys
sys.path.append("ext/booksim2/api/build")
from greenlet import greenlet
import booksim2
from ext.booksim2.api.booksim_sync import BookSim2Sync
           
def rope_rearrange_comp_air(sim: BookSim2Sync):
    """
    Rearranges and injects computation air packets into the simulation in multiple steps.

    Args:
        sim (BookSim2Sync): The simulation object to which the computation air packets will be injected.
    """
    q = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
    
    # Step 1
    sim.inject_comp_air(ca_type=4, data=q[1], t_inject=0, src=8, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=8)
    sim.inject_comp_air(ca_type=4, data=q[3], t_inject=0, src=9, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=8)
    sim.inject_comp_air(ca_type=4, data=q[0], t_inject=0, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=8)
    sim.inject_comp_air(ca_type=4, data=q[2], t_inject=0, src=1, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=8)
    for i in range(5): 
        sim.run_step()
        if len(sim.info) > 4: print("[Step-1]", sim.info)
    
    # Step 2
    sim.inject_comp_air(ca_type=0, data=0, t_inject=0, src=8, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=1+8)
    sim.inject_comp_air(ca_type=0, data=0, t_inject=0, src=9, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=1+8)
    sim.inject_comp_air(ca_type=5, data=0, t_inject=0, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=0)
    sim.inject_comp_air(ca_type=5, data=0, t_inject=0, src=1, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=0)
    for i in range(5): 
        sim.run_step()
        if len(sim.info) > 4: print("[Step-2]", sim.info)
    
    # Step 3
    sim.inject_comp_air(ca_type=5, data=0, t_inject=0, src=8, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=0)
    sim.inject_comp_air(ca_type=5, data=0, t_inject=0, src=9, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=0)
    sim.inject_comp_air(ca_type=4, data=q[5], t_inject=0, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=8)
    sim.inject_comp_air(ca_type=4, data=q[7], t_inject=0, src=1, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=8)
    for i in range(5): 
        sim.run_step()
        if len(sim.info) > 4: print("[Step-3]", sim.info)
    
    # Step 4
    sim.inject_comp_air(ca_type=4, data=q[4], t_inject=0, src=8, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=8)
    sim.inject_comp_air(ca_type=4, data=q[6], t_inject=0, src=9, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=8)
    sim.inject_comp_air(ca_type=0, data=0, t_inject=0, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=1+8)
    sim.inject_comp_air(ca_type=0, data=0, t_inject=0, src=1, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=1+8)
    for i in range(5): 
        sim.run_step()
        if len(sim.info) > 4: print("[Step-4]", sim.info)
    
    # Step 5
    sim.inject_comp_air(ca_type=5, data=0, t_inject=0, src=8, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=0)
    sim.inject_comp_air(ca_type=5, data=0, t_inject=0, src=9, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=0)
    sim.inject_comp_air(ca_type=5, data=0, t_inject=0, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=0)
    sim.inject_comp_air(ca_type=5, data=0, t_inject=0, src=1, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=0)
    for i in range(5): 
        sim.run_step()
        if len(sim.info) > 4: print("[Step-5]", sim.info)

if __name__ == "__main__":
    sim = BookSim2Sync()
    res = rope_rearrange_comp_air(sim)