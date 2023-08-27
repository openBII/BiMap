# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

include scripts/path.mak

TARGET_TYPE := STATIC_EXE
TARGET := behavior_simulator

SRCS := $(BEHAVIOR_DIR)/main.cpp \
       $(BEHAVIOR_DIR)/context.cpp \
       $(BEHAVIOR_DIR)/data_block.cpp \
       $(BEHAVIOR_DIR)/identity.cpp \
       $(BEHAVIOR_DIR)/noc.cpp \
       $(BEHAVIOR_DIR)/packet.cpp \
       $(BEHAVIOR_DIR)/virtual_memory.cpp \
       $(BEHAVIOR_DIR)/patch.cpp \
       $(PRIM_DIR)/prim_02.cpp \
       $(PRIM_DIR)/prim_03.cpp \
       $(PRIM_DIR)/prim_04.cpp \
       $(PRIM_DIR)/prim_05.cpp \
       $(PRIM_DIR)/prim_06.cpp \
       $(PRIM_DIR)/prim_07.cpp \
       $(PRIM_DIR)/prim_08.cpp \
       $(PRIM_DIR)/prim_25.cpp \
       $(PRIM_DIR)/prim_26.cpp \
       $(PRIM_DIR)/prim_41.cpp \
       $(PRIM_DIR)/prim_43.cpp \
       $(PRIM_DIR)/prim_81.cpp \
       $(PRIM_DIR)/prim_83.cpp \
       $(PRIM_DIR)/primitive.cpp \
       $(UTIL_DIR)/file_utils.cpp \
       $(THIRD_PARTY_DIR)/json/json_reader.cpp \
       $(THIRD_PARTY_DIR)/json/json_value.cpp \
       $(THIRD_PARTY_DIR)/json/json_writer.cpp \
       $(IR_DIR)/basic.pb.cc \
       $(IR_DIR)/asm.pb.cc \
       $(IR_DIR)/msg.pb.cc \
       $(IO_DIR)/client.cpp \
       $(IR_DIR)/data.pb.cc \
       $(TOP_DIR)/global_config.cpp 

INCS += $(THIRD_PARTY_DIR)

include scripts/rules.mak
       
install:
	cp behavior_simulator ../old/generator/test/1C_1P/simulator
