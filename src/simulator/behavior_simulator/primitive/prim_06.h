// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef PRIM_06_H
#define PRIM_06_H

#include "src/simulator/behavior_simulator/primitive/primitive.h"

struct Prim06_Parameter : public Parameter {
    int32_t length_in;
    int32_t length_out;
    int32_t length_ciso;
    int32_t num_in;
    int32_t num_out;
    int32_t num_ciso;
    int32_t x1_precision;
    int32_t out_precision;
    int32_t bit_shift_num;
    bool real_length_in_en;
    int32_t real_num_in;
};

class Prim06 : public Primitive
{
 private:
    int32_t Km_num_in;
    int32_t Km_num_ciso;
    int32_t length_in_equal;
    int32_t length_ciso_equal;
    int32_t num_in_4B;
    int32_t Km_num_out;
    int32_t length_out_equal;

 public:
    Prim06(shared_ptr<Prim06_Parameter> para);
    vector<vector<int32_t>> get_output_shape() const override;
    void execute(const vector<DataBlock> &input,
                 vector<DataBlock> &output) const override;

 private:
    shared_ptr<int32_t> execute(int32_t **array_in, int32_t **array_ciso) const;
};

#endif  // PRIM_06_H
