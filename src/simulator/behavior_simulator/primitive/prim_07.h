// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef PRIM_07_H
#define PRIM_07_H

#include "src/simulator/behavior_simulator/primitive/primitive.h"

struct Prim07_Parameter : public Parameter {
    int32_t group_num;
    int32_t neuron_real_num;
    int32_t lut_data_width;
    int32_t x1_precision;  // 0:int32 1/2/3 int8
    int32_t x2_precision;  // 0:int32 1/2/3 int8
    int32_t bit_shift_num;
};

class Prim07 : public Primitive
{
 private:
    int32_t neuron_num;
    int32_t neuron_real_num_wr;
    int32_t x_num_in_4B;
    int32_t LUT_length;
    int32_t y_num_in_4B;
    int32_t Y_neuron_num_wr;

 public:
    Prim07(shared_ptr<Prim07_Parameter> para);
    vector<vector<int32_t>> get_output_shape() const override;
    void execute(const vector<DataBlock> &input,
                 vector<DataBlock> &output) const override;

 private:
    shared_ptr<int32_t> execute(int32_t **array_in, int32_t *array_lut) const;
};

#endif  // PRIM_07_H
