import sys
sys.path.append("3rdParties/booksim2/api/build")
from greenlet import greenlet
import booksim2
import queue

MAX_QUEUE = 16

class BookSim2:
    
    # init
    def __init__(self):
        self.gr = greenlet(self.task)
        self.dum = greenlet(self.dummy)
        self.i_fifo = queue.Queue(MAX_QUEUE)
        self.o_fifo = queue.Queue(MAX_QUEUE)  
    
    def run(self, msg = None):
        if msg != None: self.i_fifo.put(msg)
        self.gr.switch()
    
    def dummy(self): ...
        
    def task(self):
        
        # Init ...
        bsim2 = booksim2.booksim()
        bsim2.prepare("3rdParties/booksim2/src/examples/mesh88_workload_api.config")

        for i in range(10):
            
            # Push Memory
            self.dum.switch()
            
            # Get Memory
            if self.i_fifo.empty() == False: print(self.i_fifo.get())
            
            # Enject Round ...
            print("Task: step", i)
            if i == 0: bsim2.run()
            if i == 1: bsim2.end()
            
            # Push Results
            self.o_fifo.put("O-FIFO {}".format(i))
        
if __name__ == "__main__":
    sim = BookSim2()
    for i in range(3):
        sim.run()
        if sim.o_fifo.empty() == False:
            print(sim.o_fifo.get())