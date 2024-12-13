import sys
sys.path.append("3rdParties/booksim2/api/build")
from greenlet import greenlet
import booksim2

class BookSim2:
    ...
    
def task1():
    print("Task 1: Start")
    gr2.switch()  # 切换到 task2
    print("Task 1: Resume")
    msg = "hello"
    gr2.switch(msg)  # 再次切换到 task2
    print("Task 1: End")
 
def task2():
    print("Task 2: Start")
    rgr = gr1.switch()
    print("Task 2: Resume", rgr)
    gr1.switch()
    print("Task 2: End")

if __name__ == "__main__":
    gr1 = greenlet(task1)
    gr2 = greenlet(task2)
    gr1.switch()
    booksim2.run_sim("3rdParties/booksim2/src/examples/mesh88_workload_api.config")
    print("***** Run booksim done *****")
    
