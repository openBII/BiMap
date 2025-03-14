import os
import mmap
import sys
sys.path.append("ext/booksim2/api/build")
from greenlet import greenlet
import booksim2
from ext.booksim2.api.booksim_sync import BookSim2Sync

def scalar_comp_air(sim: BookSim2Sync, x: float, op: int):
    # ac op= x
    if op <= 1:
        sim.inject_comp_air(ca_type=4, data=0, t_inject=0, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=1+4+8)
    else:
        sim.inject_comp_air(ca_type=4, data=1, t_inject=0, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=1+4+8)
    sim.inject_comp_air(ca_type=0, data=x, t_inject=0, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=op)
    res_token = ""
    # Run the simulation for 5 steps to collect results
    for i in range(5): 
        sim.run_step()
        # Check if the simulation info has enough data
        if len(sim.info) < 2: return res_token
        else: res_token = sim.info
    return res_token

def scalar_r_comp_air(sim: BookSim2Sync, x: float, op: int):
    # x op= ac
    sim.inject_comp_air(ca_type=4, data=x, t_inject=0, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=1+4+8)
    if op <= 1:
        sim.inject_comp_air(ca_type=0, data=0, t_inject=0, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=op)
    else:
        sim.inject_comp_air(ca_type=0, data=1, t_inject=0, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=op)
    res_token = ""
    # Run the simulation for 5 steps to collect results
    for i in range(5): 
        sim.run_step()
        # Check if the simulation info has enough data
        if len(sim.info) < 2: return res_token
        else: res_token = sim.info
    return res_token