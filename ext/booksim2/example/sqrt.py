import os
import mmap
import sys
sys.path.append("ext/booksim2/api/build")
from greenlet import greenlet
import booksim2
from ext.booksim2.api.booksim_sync import BookSim2Sync
           
def sqrt_comp_air(sim: BookSim2Sync, x: float, iter_num: int, is_first: bool = True):
    """
    Perform a square root computation using the BookSim2 simulator.

    Args:
        sim (BookSim2Sync): An instance of the BookSim2Sync simulator.
        x (float): The number for which the square root is to be calculated.
        iter_num (int): The number of iterations to perform the computation.
        is_first (bool, optional): A flag indicating if this is the first computation. Defaults to True.

    Returns:
        str: The result token obtained from the simulation.
    """
    # Inject computation air packet with c (c[0] = 1) to initialize the simulation
    c = 1
    if is_first: sim.inject_comp_air(ca_type=4, data=2, t_inject=0, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=1, op_0=8)
    
    for i in range(iter_num):
        for j in range(1): sim.run_step()
        sim.inject_comp_air(ca_type=4, data=c, t_inject=0, src=8, iter_tag=0, pkg_size=1, x_0=0, y_0=-1, op_0=8)
        for j in range(2): sim.run_step()
        # Inject computation air packet to perform the main computation
        sim.inject_comp_air(ca_type=0, data=1, t_inject=0, src=0, iter_tag=(iter_num-1), pkg_size=1,   
                                                                    x_0=0, y_0=0, op_0=3, # /= C
                                                                    x_1=0, y_1=0, op_1=0, # += C
                                                                    x_2=0, y_2=1, op_2=3) # /= 2
        c = (c + x / c) / 2
    
    res_token = ""
    # Run the simulation for 3 steps to collect results
    for i in range(3): 
        sim.run_step()
        # Check if the simulation info has enough data
        if len(sim.info) < 2: return res_token
        else: res_token = sim.info
    return res_token

if __name__ == "__main__":
    sim = BookSim2Sync()
    res = sqrt_comp_air(sim, 2, 4)
    sim.run_step(end=True)
    print("[***] sqrt", res)