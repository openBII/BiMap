add_library(ramulator-dram OBJECT)

target_sources(
  ramulator-dram PRIVATE
  dram.h  node.h  spec.h  lambdas.h  
  
  lambdas/preq.h  lambdas/rowhit.h  lambdas/rowopen.h 

  impl/HBM3.cpp
  impl/HBM3-PIM.cpp
)

target_link_libraries(
  ramulator
  PRIVATE
  ramulator-dram
)
