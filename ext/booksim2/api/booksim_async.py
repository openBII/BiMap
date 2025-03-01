import sys
sys.path.append("ext/booksim2/api/build")
from greenlet import greenlet
import booksim2

class BookSim2Async:
    
    def __init__(self):
        self.gr = greenlet(self.task)
        self.dum = greenlet(self.dummy)
        self.bsim2 = booksim2.booksim()
    
    def run(self): self.gr.switch()
    
    def dummy(self): ...
    
    def inject(self, src, dst, t_inject):
        self.bsim2.inject(0, 10, i)
        
    def task(self):
        # Init ...
        self.bsim2.init("ext/booksim2/src/examples/mesh88_simulate.config", False)
        for i in range(10):
            # Push Memory
            self.dum.switch()
            # Enject Round ...
            print("Task: step", i)
            if i == 1: 
                print(input(), "-> Simulation End ...")
                self.bsim2.end()
        
if __name__ == "__main__":
    sim = BookSim2Async()
    for i in range(3):
        sim.inject(0, 10, 20)
        sim.run()