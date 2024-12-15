cmake_minimum_required(VERSION 2.8.12)
project(booksim2)   

add_subdirectory(pybind11)

include_directories(./)
include_directories(../src)
include_directories(../src/allocators)
include_directories(../src/arbiters)
include_directories(../src/routers)
include_directories(../src/networks)
include_directories(../src/netrace)
include_directories(../src/power)

set(sources ../src/config_utils.cpp 
            ../src/booksim_config.cpp 
            ../src/module.cpp 
            ../src/buffer.cpp 
            ../src/vc.cpp 
            ../src/routefunc.cpp 
            ../src/traffic.cpp
            ../src/flitchannel.cpp 
            ../src/trafficmanager.cpp 
            ../src/synthetictrafficmanager.cpp
            ../src/steadystatetrafficmanager.cpp 
            ../src/batchtrafficmanager.cpp
            ../src/workloadtrafficmanager.cpp
            ../src/simulatetrafficmanager.cpp
            ../src/buffer_state.cpp 
            ../src/stats.cpp
            ../src/credit.cpp
            ../src/outputset.cpp
            ../src/flit.cpp
            ../src/injection.cpp
            ../src/misc_utils.cpp
            ../src/rng_wrapper.cpp
            ../src/workload.cpp
            ../src/rng_double_wrapper.cpp
            ../src/y.tab.c
            ../src/lex.yy.c)
            
file(GLOB_RECURSE power_srcs ../src/power/*.cpp)
file(GLOB_RECURSE router_srcs ../src/routers/*.cpp)
file(GLOB_RECURSE networks_srcs ../src/networks/*.cpp)
file(GLOB_RECURSE netrace_srcs ../src/netrace/netrace.c)
file(GLOB_RECURSE netrace_util_srcs ../src/netrace/util/*.c)
file(GLOB_RECURSE allocators_srcs ../src/allocators/*.cpp)
file(GLOB_RECURSE arbiters_srcs ../src/arbiters/*.cpp)

pybind11_add_module(booksim2 top.cpp ${sources} ${router_srcs} 
                                     ${power_srcs} ${networks_srcs} 
                                     ${netrace_srcs} ${netrace_util_srcs}
                                     ${allocators_srcs} ${arbiters_srcs})