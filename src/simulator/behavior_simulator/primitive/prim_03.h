// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef PRIM_03_H
#define PRIM_03_H

#include "src/simulator/behavior_simulator/primitive/primitive.h"

struct Prim03_Parameter : public Parameter {
    bool tensor_en;
    int32_t x1_precision;
    int32_t bias_type;
    int32_t bias_length;
    int32_t ny;
    int32_t nx;
    int32_t stride_x;
    int32_t stride_y;
    int32_t n_branch;
    int32_t nif;
    int32_t constant_b = 0;
};

class Prim03 : public Primitive
{
 private:
    int32_t cin_wr_real;
    int32_t num_in_4B;
    int32_t Ox;
    int32_t Oy;

 public:
    Prim03(shared_ptr<Prim03_Parameter> para);
    vector<vector<int32_t>> get_output_shape() const override;
    void execute(const vector<DataBlock> &input,
                 vector<DataBlock> &output) const override;

 private:
    shared_ptr<int32_t> execute(Array<int32_t, 2> &p_x1,
                                Array<int32_t, 2> &p_x2, int32_t *SB) const;
    shared_ptr<int32_t> execute(Array<int32_t, 3> &p_x1,
                                Array<int32_t, 3> &p_x2, int32_t *SB) const;
};

#endif  // PRIM_03_H
