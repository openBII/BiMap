// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef PRIM_04_H
#define PRIM_04_H

#include "src/simulator/behavior_simulator/primitive/primitive.h"

struct Prim04_Parameter : public Parameter {
    int32_t x1_precision;
    int32_t x2_precision;
    int32_t bias_type;
    int32_t nif;
    int32_t nof;
    int32_t constant_b = 0;
};

class Prim04 : public Primitive
{
 private:
    int32_t Km_num;
    int32_t length_in_equal;
    int32_t InA_num_in_4B;
    int32_t w_wr_real;
    int32_t w_grp_num;
    int32_t cout_in_grp;
    int32_t InB_num_in_4B;

 public:
    Prim04(shared_ptr<Prim04_Parameter> para);
    vector<vector<int32_t>> get_output_shape() const override;
    void execute(const vector<DataBlock> &input,
                 vector<DataBlock> &output) const override;

 private:
    shared_ptr<int32_t> execute(int32_t *X1, Array<int32_t, 2> &W,
                                int32_t *SB) const;
};

#endif  // PRIM_04_H
