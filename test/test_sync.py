from src.simulator.resource_simulator.st_env import STEnv
from src.simulator.task_rabbit.task_model.shape import Shape
from src.simulator.task_rabbit.task_model.precision import Precision
from src.simulator.task_rabbit.task_model.task_graph import TaskGraph
from src.simulator.resource_simulator.config.matrix_config import ServerConfig
from src.simulator.resource_simulator.st_model.space_matrix.server_factory import ServerFactory
from src.simulator.resource_simulator.st_model.st_coord import create_mlcoord
from src.simulator.task_rabbit.task_model.transformer import create_input, create_mlp
from src.simulator.resource_simulator.st_draw import STDraw


# Construct a task graph
task_graph = TaskGraph()
_, input_data = create_input(task_graph, Shape(batch=64, token=512, nf=2048), 
                             Precision.FLOAT_16)
output0, dict0 = create_mlp(
    task_graph, input_data, Shape(batch=64, token=512, nr=2048, nf=4096), 
    Precision.FLOAT_16, False)
output1, dict1 = create_mlp(
    task_graph, output0, Shape(batch=64, token=512, nr=4096, nf=1024), 
    Precision.FLOAT_16, True)

STDraw.draw_graph(task_graph, 
                  out_path='temp/mlp_sync.task.html',
                  width='1920px', height='1080px')

# Construct a hardware
config = ServerConfig("top/gpu_server.toml")
server = ServerFactory.create_matrix(config)

# Construct a simulation environment
st_env = STEnv(task_graph, server)
st_env.enable_pipeline()

# Define some constants for writing coordinates
CHIP = 0
DRAM = 2
PCIE_PHY = 3
TENSOR_UNIT = 1
ROUTER = 3
SHARED_MEMORY = (3, 1)
SRAM_BUFFER = 0

# Map the task graph onto the hardware
# Task Mapping
dram0 = create_mlcoord((0, 0), DRAM)
st_env.put_in(dram0, input_data.id)
st_env.put_in(dram0, dict0["weight"].id)
sync_id0 = st_env.get_sync_id()
st_env.sync(dram0, sync_id0)
st_env.put_in(dram0, dict1["weight"].id)
st_env.put_in(dram0, dict0["output"].id)
st_env.put_in(dram0, dict1["output"].id)

mac_array0 = create_mlcoord((0, 0), CHIP, (0, 0), TENSOR_UNIT)
st_env.put_in(mac_array0, dict0["compute"].id)
st_env.sync(mac_array0, sync_id0)
st_env.put_in(mac_array0, dict1["compute"].id)

# Edge Mapping
st_env.auto_edge_map()

# Simulate
st_env.simulate()
st_env.show_overall_time()
