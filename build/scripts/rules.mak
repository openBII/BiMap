# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

# phony target
.PHONY: all clean

CC=g++

INCS += $(ROOT_DIR)

CFLAGS += -g # -fmodules-ts

# get the dir of dependency libs
LIB_DIR := $(sort $(dir $(LIBS)))
# get the file name of dependency libs
LIB_LIST := $(sort $(notdir $(LIBS)))

# extract the pattern from the lib name to statisfy the -l option of gcc
LIB_LIST := $(patsubst lib%.so, %, $(LIB_LIST))

cmd_targets := clean
# specify the linking libraires
LD_LIBS := $(addprefix -I, $(INCS))
LD_LIBS := $(addprefix -l, $(LIB_LIST))
LD_LIBS += -lm -lstdc++ -lpthread -lprotobuf  -fopenmp
# LD_LIBS += -Xcompiler 

# set the temp build directory
BUILT_TMP := $(basename $(TARGET))
BUILT_TMP := $(addsuffix Temp/, $(BUILT_TMP))

# set the obj list
OBJS := $(basename $(notdir $(SRCS)))
OBJS += $(notdir $(CU_SRCS))
OBJS := $(addsuffix .o, $(OBJS))
OBJS := $(addprefix $(BUILT_TMP), $(OBJS))

# set the dependency file list
DEPS := $(patsubst %.o, %.d, $(OBJS))

# get the path of all source files
space := " "
source_path := $(dir $(SRCS))
source_path := $(sort $(source_path))
paths := $(subst $(space),;,$(source_path))
# tell the make where to find the source files
vpath %.cpp $(paths)
vpath %.cu $(paths)
vpath %.cc $(paths)
include_path := $(sort $(INCS))
include_path := $(subst $(space),;,$(include_path))
vpath %.h $(include_path)

# set the compiler to support C++20
CFLAGS += -std=c++2a
CFLAGS += -Wfatal-errors

# add the directory for linking libraires
LDFLAGS += $(addprefix -L, $(LIB_DIR))

# set the include directories
INCS += $(THIRD_PARTY_DIR)
INCS := $(addprefix -I, $(INCS))


all:$(TARGET)
	# echo $(TARGET)

$(TARGET): MAKEDIR $(OBJS)
	@echo $(include_paths)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LD_LIBS)

# create the directory for the temp build files
MAKEDIR:
	@mkdir -p $(BUILT_TMP)

# include dependency files
ifeq (,$(filter $(cmd_targets), $(MAKECMDGOALS)))
sinclude $(DEPS)
endif

# 这里.d的规则是无效的
# rules to generate dependency files

# $(BUILT_TMP)%.cu.o:  %.cu
# 	$(CC) -c $(CFLAGS) $(INCS) $< -o $@

$(BUILT_TMP)%.o: %.cpp
	$(CC) -c $(CFLAGS) $(INCS) $< -o $@ -MD 

$(BUILT_TMP)%.o: %.cc
	$(CC) -c $(CFLAGS) $(INCS) $< -o $@


clean:
	@echo $(LD_LIBS)
	@echo "make " $(TARGET) "clean"
	rm -f $(TARGET) $(OBJS) $(DEPS)
