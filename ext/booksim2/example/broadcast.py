import os
import mmap
import sys
sys.path.append("ext/booksim2/api/build")
from greenlet import greenlet
import booksim2
from ext.booksim2.api.booksim_sync import BookSim2Sync

def broadcast_comp_air(sim: BookSim2Sync, src: int, targs: int, step: int, data: float):
    """
    Perform a broadcast operation in the simulation environment.

    Args:
        sim (BookSim2Sync): The simulation object.
        src (int): The source node for the broadcast.
        targs (int): The number of target nodes, must be 2, 4, or 8.
        step (int): The step size for calculating target node coordinates.
        data (float): The data to be broadcast.
    """
    # Ensure that the number of target nodes is either 2, 4, or 8
    assert targs == 2 or targs == 4 or targs == 8
    if targs == 2:
        # Inject computation air packets for 2 target nodes
        sim.inject_comp_air(ca_type=3, data=data, t_inject=1,
                            src=src, iter_tag=0, pkg_size=1,
                            x_0=step * 0, y_0=0, op_0=0+8,
                            x_1=step * 1, y_1=0, op_1=0+8)
    elif targs == 4:
        # Inject computation air packets for 4 target nodes
        sim.inject_comp_air(ca_type=3, data=data, t_inject=1,
                            src=src, iter_tag=0, pkg_size=1,
                            x_0=step * 0, y_0=0, op_0=0+8,
                            x_1=step * 1, y_1=0, op_1=0+8,
                            x_2=step * 2, y_2=0, op_2=0+8)
    else:
        # Inject computation air packets for 8 target nodes
        sim.inject_comp_air(ca_type=3, data=data, t_inject=1, 
                            src=src, iter_tag=0, pkg_size=1,
                            x_0=step * 0, y_0=0, op_0=0+8,
                            x_1=step * 1, y_1=0, op_1=0+8,
                            x_2=step * 2, y_2=0, op_2=0+8,
                            x_3=step * 3, y_3=0, op_3=0+8)
    # Run the simulation for 5 steps
    for i in range(4):
        sim.run_step()

if __name__ == "__main__":
    sim = BookSim2Sync()
    broadcast_comp_air(sim, src=0, targs=8, step=4, data=2.5)
    sim.run_step(end=True) # End