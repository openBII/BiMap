// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef PRIM_25_H
#define PRIM_25_H

#include "src/simulator/behavior_simulator/primitive/primitive.h"

struct Prim25_Parameter : public Parameter {
    int32_t x1_precision;
    int32_t out_precision;
    int32_t nif;
    int32_t nof;
    int32_t nix;
    int32_t niy;
    int32_t pad_top;
    int32_t pad_down;
    int32_t pad_left;
    int32_t pad_right;
    int32_t nkx;
    int32_t nky;
    int32_t stride_x;
    int32_t stride_y;
    int32_t compare_init;
    int32_t bit_shift_num;
};

class Prim25 : public Primitive
{
 private:
    int32_t Km_num_in;
    int32_t cin_wr_real;
    int32_t Px_real;
    int32_t Py_real;
    int32_t Output_fm_Ox;
    int32_t Output_fm_Oy;
    int32_t num_in_4B;
    int32_t Km_num_out;
    int32_t cout_real;

 public:
    Prim25(shared_ptr<Prim25_Parameter> para);
    vector<vector<int32_t>> get_output_shape() const override;
    void execute(const vector<DataBlock> &input,
                 vector<DataBlock> &output) const override;

 private:
    shared_ptr<int32_t> execute(int32_t ***SI) const;
};

#endif  // PRIM_25_H
