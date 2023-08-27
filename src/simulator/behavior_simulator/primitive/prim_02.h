// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef PRIM_02_H
#define PRIM_02_H

#include "src/simulator/behavior_simulator/primitive/primitive.h"

struct Prim02_Parameter : public Parameter {
    int32_t x1_precision;
    int32_t bias_type;
    int32_t bias_length;
    int32_t nif;
    int32_t nix;
    int32_t niy;
    bool has_pad;
    int32_t pad_top;
    int32_t pad_down;
    int32_t pad_left;
    int32_t pad_right;
    int32_t nkx;
    int32_t nky;
    int32_t stride_x;
    int32_t stride_y;
    int32_t constant_b;
    bool avg_pooling_en;
};

class Prim02 : public Primitive
{
 private:
    int32_t Km_num;
    int32_t cin_wr_real;
    int32_t Px_real;
    int32_t Py_real;
    int32_t array_num;
    int32_t Output_fm_Ox;
    int32_t Output_fm_Oy;
    int32_t num_in_4B;

 public:
    Prim02(shared_ptr<Prim02_Parameter> para);
    vector<vector<int32_t>> get_output_shape() const override;
    void execute(const vector<DataBlock> &input,
                 vector<DataBlock> &output) const override;

 private:
    // shared_ptr<int32_t> execute(void *SI, int32_t *SB) const;
    shared_ptr<int32_t> execute(Array<int32_t, 3> &,
                                int32_t *) const; /**< pooling*/
    shared_ptr<int32_t> execute(Array<int32_t, 4> &,
                                int32_t *) const; /**< add*/
    void mem2si(uint32_t *p_mem_si, Array<int32_t, 3> &) const;
    void mem2si(uint32_t *p_mem_si, Array<int32_t, 4> &) const;
    void pad(Array<int32_t, 3> &SI) const;
};

#endif  // PRIM_02_H
