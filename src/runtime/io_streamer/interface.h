// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#pragma once

#include <fstream>

class ICore{
    public:
        virtual void setMemory(int addr, int len, FILE* f) = 0;
        virtual void setMemory(int addr, int len, char *buf) = 0;
        virtual int getMemory(int addr) = 0;

};
class IChip{
    public:
        virtual ICore* getCore(int x, int y) = 0;
};

class IChipArray{
    public:
        virtual IChip* getChip(int x, int y) = 0;
};