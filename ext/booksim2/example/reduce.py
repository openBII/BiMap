import os
import mmap
import sys
sys.path.append("ext/booksim2/api/build")
from greenlet import greenlet
import booksim2
from ext.booksim2.api.booksim_sync import BookSim2Sync

def reduce_comp_air(sim: BookSim2Sync, sources: int, step: int, data: float, para_num: int = 1, op: int = 0):
    """
    Perform a reduction operation using computation air packets in the simulation.

    Args:
        sim (BookSim2Sync): The simulation object.
        sources (int): The number of source nodes, must be 4, 8, or 16.
        step (int): The step size for calculating node coordinates.
        data (float): The data to be used in the reduction operation.
        op (int, optional): The operation code. Defaults to 0.

    Returns:
        str: The result token obtained from the simulation info.
    """
    # Ensure that the number of source nodes is either 4, 8, or 16
    assert sources == 16 or sources == 8 or sources == 4
    paral_offset = [0,1,8,9]
    
    # Initialize a list to store reduction counter
    node = []
    for i in range(sources):
        # Calculate the node coordinate and append it to the list
        node.append((i%4)*step + 8*(i//4)*step)
        # Inject computation air packets to initialize nodes
        if i * 2 != sources:
            for j in range(para_num):
                sim.inject_comp_air(ca_type=4, data=2, t_inject=0, src=paral_offset[j], iter_tag=0, pkg_size=1, x_0=(i%4)*step, y_0=(i//4)*step, op_0=1+4+8)
                
    # Run the simulation for the number of source nodes
    for i in range(sources * para_num): 
        sim.run_step()
    # Initialize a list to store the path for each source node
    path = []
    for i in range(sources):
        
        path.append([])
        
        # Level 0:
        if i % 2 == 0: path[i].append((step, 0))
        else: path[i].append((0, 0))
        # Level 1:
        if sources > 4:
            if i % 4 < 2: path[i].append((step, 0))
            else: path[i].append((-1 * step, 0))
        # Level 2:
        if sources == 16:
            if i % 8 < 4: path[i].append((-2*step, step))
            else: path[i].append((-2*step, 0))
        
    # Level Final:
    for i in range(sources):
        if i < sources // 2: 
            if sources == 4: path[i].append((-1 * step, 0))
            if sources == 8: path[i].append((-2 * step, 0))
            if sources == 16: path[i].append((0, -1 * step))
        else: 
            if sources == 4: path[i].append((-3 * step, 0))
            if sources == 8: path[i].append((-2 * step, -1 * step))
            if sources == 16: path[i].append((0, -3 * step))
    
    # print("[path]", path)
    # print("[node]", node)
    
    for i in range(sources):
        for j in range(para_num):
            # Inject computation air packets based on the number of source nodes
            if sources == 4:
                sim.inject_comp_air(ca_type=1, data=data, t_inject=0, src=node[i]+paral_offset[j], iter_tag=0, pkg_size=1,   
                                                                    x_0=path[i][0][0], y_0=path[i][0][1], op_0=op+8, 
                                                                    x_1=path[i][1][0], y_1=path[i][1][1], op_1=op+8)
            elif sources == 8:
                sim.inject_comp_air(ca_type=1, data=data, t_inject=0, src=node[i]+paral_offset[j], iter_tag=0, pkg_size=1,   
                                                                    x_0=path[i][0][0], y_0=path[i][0][1], op_0=op+8, 
                                                                    x_1=path[i][1][0], y_1=path[i][1][1], op_1=op+8,
                                                                    x_2=path[i][2][0], y_2=path[i][2][1], op_2=op+8)
            elif sources == 16:
                sim.inject_comp_air(ca_type=1, data=data, t_inject=0, src=node[i]+paral_offset[j], iter_tag=0, pkg_size=1,   
                                                                    x_0=path[i][0][0], y_0=path[i][0][1], op_0=op+8, 
                                                                    x_1=path[i][1][0], y_1=path[i][1][1], op_1=op+8,
                                                                    x_2=path[i][2][0], y_2=path[i][2][1], op_2=op+8,
                                                                    x_3=path[i][3][0], y_3=path[i][3][1], op_3=op+8)
    
    # Initialize the result token
    res_token = ""
    # Run the simulation for a few more steps to collect results
    for i in range(sources*para_num+4): 
        sim.run_step()
        # Check if the simulation info has enough data
        if len(sim.info) < 2: return res_token
        else: res_token = sim.info
        
if __name__ == "__main__":
    
    sim = BookSim2Sync()
    res = reduce_comp_air(sim, sources=16, step=2, data=2.5, para_num=4)
    print("[***]", res)
    
    """
    
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
    """