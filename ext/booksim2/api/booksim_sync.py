import sys
sys.path.append("ext/booksim2/api/build")
from greenlet import greenlet
import booksim2

class BookSim2Sync:
    
    def __init__(self, max_round = 20):
        self.max_round = max_round
        self.gr = greenlet(self.task)
        self.dum = greenlet(self.dummy)
        self.bsim2 = booksim2.booksim()
        self.end = False
    
    def run_step(self, end = False): 
        self.end = end
        self.gr.switch()
    
    def dummy(self): ...
    
    def inject(self, delay, src, dst, size=1):
        self.bsim2.inject(src, dst, delay, size)
        
    def task(self):
        # Init ...
        self.bsim2.init("ext/booksim2/src/examples/mesh88_simulate.config", True)
        for i in range(self.max_round):
            # Push Memory
            self.dum.switch()
            # Enject Round ...
            print("\n[*] Task: step", i + 1)
            self.bsim2.run_sync()
            self.bsim2.eject_all_print() # Print Ejection Results
            if self.end or i + 1 >= self.max_round: self.bsim2.end()
        
if __name__ == "__main__":
    sim = BookSim2Sync()
    # ----------------------------------------
    sim.inject(0, 10, 20)       # 0 + 0 = 0
    sim.inject(10, 3, 26)       # 0 + 10 = 10
    sim.inject(11, 3, 28)       # 10 + 11 = 21
    sim.inject(12, 5, 32)       # 21 + 12 = 33
    for i in range(5): sim.run_step()
    # ----------------------------------------
    sim.inject(1112, 5, 32, 3)  # 33 + 1112 = 1145
    for i in range(1): sim.run_step()
    # ----------------------------------------
    sim.inject(2112, 6, 43, 1)  # 1145 + 2112 = 3246
    sim.inject(1000, 24, 54, 1) # 3246 + 1000 = 4246
    for i in range(2): sim.run_step()
    # ----------------------------------------
    sim.inject(2000, 24, 54, 1) # 4246 + 2000 = 6246
    for i in range(2): sim.run_step()
    # ----------------------------------------
    sim.run_step(end=True) # End
    