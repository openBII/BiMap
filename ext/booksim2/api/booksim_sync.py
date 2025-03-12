import os
import mmap
import sys
sys.path.append("ext/booksim2/api/build")
from greenlet import greenlet
import booksim2

class BookSim2Sync:
    
    def __init__(self, max_round = 400):
        self.max_round = max_round
        self.gr = greenlet(self.task)
        self.dum = greenlet(self.dummy)
        self.bsim2 = booksim2.booksim()
        self.end = False
        self.info = ""
    
    def run_step(self, end = False): 
        self.end = end
        self.gr.switch()

    def run_ahead(self, end=False):
        print("run_ahead is called", os.getpid())
        shared_mem_size = 4096
        shared_mem = mmap.mmap(-1, shared_mem_size)
        shared_mem.seek(0)
        shared_mem.write(b'\x00')

        ret = os.fork()
        if ret == 0:
            try:
                print("Child Process Bgn", os.getpid())
                info_res = ""
                cnt = 0
                while cnt < 3:
                    self.run_step()
                    if self.info == "":
                        cnt += 1
                    else:
                        info_res += self.info
                shared_mem.seek(1)
                shared_mem.write(info_res.encode())
                shared_mem.seek(0)
                shared_mem.write(b'\x01')
                print("Child Process End", os.getpid())
            finally:
                shared_mem.close()
            exit(0)
        else:
            try:
                print("Parent Process Bgn", os.getpid())
                os.waitpid(ret, 0)
                shared_mem.seek(0)
                if shared_mem.read(1) == b'\x01':
                    shared_mem.seek(1)
                    info_res = shared_mem.read(shared_mem_size - 1).decode().rstrip('\x00')
                else:
                    info_res = ""
                print("Parent Process End", os.getpid())
                return info_res
            finally:
                shared_mem.close()
         
    def dummy(self): ...
    
    def inject(self, delay, src, dst, size=1):
        self.bsim2.inject(src, dst, delay, size)
        
    def inject_comp_air(self, ca_type: int, 
                        data: float, t_inject: int,
                        src: int, iter_tag: int, pkg_size: int,
                        x_0: int = -1, y_0: int = -1, op_0: int = -1, 
                        x_1: int = -1, y_1: int = -1, op_1: int = -1, 
                        x_2: int = -1, y_2: int = -1, op_2: int = -1, 
                        x_3: int = -1, y_3: int = -1, op_3: int = -1):
        self.bsim2.inject_comp_air(ca_type, data, t_inject, src, iter_tag, pkg_size, 
                                   x_0, y_0, op_0, x_1, y_1, op_1, 
                                   x_2, y_2, op_2, x_3, y_3, op_3)
        
    def task(self):
        # Init ...
        self.bsim2.init("ext/booksim2/src/examples/mesh88_simulate.config", True)
        for i in range(self.max_round):
            # Push Memory
            self.dum.switch()
            # Enject Round ...
            self.info = ""
            # print("\n[*] Task: step", i + 1)
            self.bsim2.run_sync()
            self.info = self.bsim2.eject_all_print()
            if self.end or i + 1 >= self.max_round:
                print("Booksim End", self.end, i)
                self.bsim2.end()
        
if __name__ == "__main__":
    sim = BookSim2Sync()
    # ----------------------------------------
    # sim.inject(0, 10, 20) # Stat from the last end (0)
    # sim.inject(0, 3, 26) # 3
    # sim.inject(0, 3, 28) # 13
    # sim.inject(0, 4, 32) # 24
    # manual config
    sim.inject_comp_air(ca_type=4, data=3.0, t_inject=1, src=6, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=8)
    sim.inject_comp_air(ca_type=4, data=3.1, t_inject=1, src=6, iter_tag=0, pkg_size=1, x_0=0, y_0=1, op_0=8)
    sim.inject_comp_air(ca_type=4, data=3.2, t_inject=1, src=6, iter_tag=0, pkg_size=1, x_0=-1, y_0=1, op_0=8)
    sim.inject_comp_air(ca_type=4, data=2.5, t_inject=1, src=5, iter_tag=0, pkg_size=1, x_0=2, y_0=0, op_0=8)
    # compute
    sim.inject_comp_air(ca_type=0, data=1.3, t_inject=1, src=5, iter_tag=0, pkg_size=1, x_0=2, y_0=0, op_0=0)
    sim.inject_comp_air(ca_type=0, data=3.3, t_inject=1, src=6, iter_tag=5, pkg_size=1, x_0=0, y_0=0, op_0=0, x_1=0, y_1=1, op_1=0, x_2=-1, y_2=0, op_2=0)
    
    for i in range(10): 
        sim.run_step()
        print(sim.info)
    
    # sim.inject(0, 5, 32, 3) # Stat from the last end (79)
    # sim.inject(0, 5, 32, 3) # Stat from the last end (79)
    
    # print("Hello2")
    # print("[run_ahead]", sim.run_ahead())

    # ----------------------------------------
    # sim.inject(0, 5, 32, 3) # Stat from the last end (79)
    # sim.inject(0, 5, 32, 3) # Stat from the last end (79)
    # for i in range(1): 
    #     sim.run_step()
    #     print(sim.info)
    # ----------------------------------------
    # sim.inject(0, 6, 43, 1) # Stat from the last end (1236)
    # sim.inject(0, 24, 54, 1) # 1236 + 2112 = 3348
    # for i in range(2): 
    #     sim.run_step()
    #     print(sim.info)
    # ----------------------------------------
    # sim.inject(0, 24, 54, 1) # Stat from the last end (4391)
    # for i in range(2): 
    #     sim.run_step()
    #     print(sim.info)
    # ----------------------------------------
    sim.run_step(end=True) # End
    