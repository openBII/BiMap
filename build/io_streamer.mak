# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

include scripts/path.mak

TARGET_TYPE := STATIC_EXE
TARGET := launch_server

SRCS := $(IO_DIR)/launch.cpp \
        $(IO_DIR)/server.cpp \
        $(IR_DIR)/msg.pb.cc


include scripts/rules.mak
       
