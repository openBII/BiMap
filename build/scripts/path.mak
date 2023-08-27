# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

ROOT_DIR := ..
BUILD_DIR := $(ROOT_DIR)/build/
TEST_DIR := $(ROOT_DIR)/test
THIRD_PARTY_DIR := $(ROOT_DIR)/3rdParties
TOP_DIR := $(ROOT_DIR)/top


COMPILER_DIR := $(ROOT_DIR)/src/compiler
SIMULATOR_DIR := $(ROOT_DIR)/src/simulator
RUNTIME_DIR := $(ROOT_DIR)/src/runtime
UTIL_DIR := $(ROOT_DIR)/src/utils

GENERATOR_DIR := $(COMPILER_DIR)/code_generator
ASSEMBLER_DIR := $(COMPILER_DIR)/assembler

BEHAVIOR_DIR := $(SIMULATOR_DIR)/behavior_simulator
CLOCK_DIR := $(SIMULATOR_DIR)/clock_simulator

PRIM_DIR := $(BEHAVIOR_DIR)/primitive
IR_DIR := $(COMPILER_DIR)/ir
IO_DIR := $(RUNTIME_DIR)/io_streamer