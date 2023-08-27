// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef PRIM_08_H
#define PRIM_08_H

#include "src/simulator/behavior_simulator/primitive/primitive.h"

struct Prim08_Parameter : public Parameter {
    int32_t neuron_num;
    int32_t group_num;
    int32_t seed;
    int32_t Vth0;
    int32_t Vth_adpt_en;
    int32_t Vth_alpha;
    int32_t Vth_beta;
    int32_t Vth_Incre;
    int32_t VR;
    int32_t VL;
    int32_t Vleaky_adpt_en;
    int32_t Vleaky_alpha;
    int32_t Vleaky_beta;
    int32_t dV;
    int32_t Ref_len;
    int32_t Tw_cnt;
    int32_t Vinit;
    int32_t Tw_len;
    int32_t Tw_en;
    int32_t VM_const_en;
    int32_t VM_const;
    int32_t VM_len;
    int32_t Vtheta_const_en;
    int32_t Vtheta_const;
    int32_t Vtheta_len;
    int32_t ref_cnt_const_en;
    int32_t ref_cnt_const;
    int32_t reset_mode;
    int32_t fire_type;
    int32_t bit_shift_num;
};

class lfsr
{
 public:
    lfsr(int32_t seed);
    int64_t update();

 private:
    int64_t _lfsr;
};

class Prim08 : public Primitive
{
 private:
    int32_t neuron_length;
    mutable lfsr _lfsr;

 public:
    Prim08(shared_ptr<Prim08_Parameter> para);
    vector<vector<int32_t>> get_output_shape() const override;
    void execute(const vector<DataBlock> &input,
                 vector<DataBlock> &output) const override;

 private:
    vector<shared_ptr<int32_t>> execute(int32_t *VM, int32_t *Vtheta,
                                        int32_t *V, int32_t *ref_cnt,
                                        int32_t *Uin) const;
};

#endif  // PRIM_08_H
