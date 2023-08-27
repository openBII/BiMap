// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef PRIM_81_H
#define PRIM_81_H

#include "src/simulator/behavior_simulator/primitive/primitive.h"

struct Prim81_Parameter : public Parameter {
    int32_t x1_precision;
    int32_t x2_precision;
    int32_t bias_type;
    int32_t niy;        // vertical length of the input
    int32_t nix;        // horizontal length of the input
    int32_t nif;        // channel number of the input
    int32_t nof;        // channel number of the output
    int32_t nky;        // vertical size of kernel
    int32_t nkx;        // horizontal size of kernel
    int32_t stride_x;   // horizontal stride length
    int32_t stride_y;   // vertical stride length
    int32_t pad_top;    // padding above
    int32_t pad_down;   // padding down
    int32_t pad_left;   // left padding
    int32_t pad_right;  // right padding
    int32_t dilate_x;   // horizontal dilatation rate
    int32_t dilate_y;   // vertical dilatation rate
};

class Prim81 : public Primitive
{
 private:
    int32_t Px_real;
    int32_t Py_real;
    int32_t Kx_real;
    int32_t Ky_real;
    int32_t Output_fm_Ox;
    int32_t Output_fm_Oy;
    int32_t w_grp_num;
    int32_t cout_wr_real;
    int32_t nix_grp_real;
    int32_t nix_wr_real;
    int32_t InA_num_in_4B;
    int32_t InB_num_in_4B;
    int32_t InB_per32cout_in_bytes;

 public:
    Prim81(shared_ptr<Prim81_Parameter> para);
    vector<vector<int32_t>> get_output_shape() const override;
    void execute(const vector<DataBlock> &input,
                 vector<DataBlock> &output) const override;

 private:
    shared_ptr<int32_t> execute(Array<int32_t, 3> &SI, Array<int32_t, 4> &SW,
                                int32_t *SB) const;
};

#endif  // PRIM_81_H
