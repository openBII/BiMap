import sys
sys.path.append("3rdParties/booksim2/api/build")
from greenlet import greenlet
import booksim2

class BookSim2:
    
    def __init__(self):
        self.gr = greenlet(self.task)
        self.dum = greenlet(self.dummy)
        self.bsim2 = booksim2.booksim()
    
    def run(self): self.gr.switch()
    
    def dummy(self): ...
    
    def inject(self, src, dst, t_inject):
        self.bsim2.inject(src, dst, t_inject)
        
    def task(self):
        # Init ...
        self.bsim2.init("3rdParties/booksim2/src/examples/mesh88_simulate.config", True)
        for i in range(10):
            # Push Memory
            self.dum.switch()
            # Enject Round ...
            print("\n[*] Task: step", i)
            if i < 3: self.bsim2.run_sync()
            elif i == 3: self.bsim2.end()
            else: print("Execution End ...")
        
if __name__ == "__main__":
    sim = BookSim2()
    for i in range(6):
        sim.inject(0, 10, 20)
        sim.run()