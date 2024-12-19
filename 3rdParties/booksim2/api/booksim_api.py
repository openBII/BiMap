import sys
sys.path.append("3rdParties/booksim2/api/build")
from greenlet import greenlet
import booksim2

class BookSim2:
    
    def __init__(self):
        self.gr = greenlet(self.task)
        self.dum = greenlet(self.dummy)
    
    def run(self): self.gr.switch()
    
    def dummy(self): ...
        
    def task(self):
        
        # Init ...
        bsim2 = booksim2.booksim()
        bsim2.init("3rdParties/booksim2/src/examples/mesh88_simulate.config")

        for i in range(10):
            
            # Push Memory
            self.dum.switch()
            
            # Enject Round ...
            print("Task: step", i)
            if i == 1: 
                print(input(), "-> Simulation End ...")
                bsim2.end()
            
            # Enject API
            bsim2.inject(0, 10, i)
        
if __name__ == "__main__":
    sim = BookSim2()
    for i in range(3):
        sim.run()