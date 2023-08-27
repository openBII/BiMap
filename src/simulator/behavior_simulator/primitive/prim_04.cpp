// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "prim_04.h"
#include "src/simulator/behavior_simulator/util.h"
#include <cmath>
#include <fstream>
#include <iomanip>

#define P04_DEBUG 0

Prim04::Prim04(shared_ptr<Prim04_Parameter> para) : Primitive(AXON, para) {
    if (para->x1_precision == 2 || para->x1_precision == 1) {
        Km_num = ceil(double(para->nif) / 16.0);
        length_in_equal = Km_num * 16;
        InA_num_in_4B = 4;
    } else {
        Km_num = ceil(double(para->nif) / 64.0);
        length_in_equal = Km_num * 64;
        InA_num_in_4B = 16;
    }

    if (para->x2_precision == 2 || para->x2_precision == 1) {
        w_grp_num = ceil(double(para->nof) / 32.0);
        w_wr_real = w_grp_num * 32;
        cout_in_grp = 32;
        InB_num_in_4B = 4;
    } else {
        w_grp_num = ceil(double(para->nof) / 128.0);
        w_wr_real = w_grp_num * 128;
        cout_in_grp = 128;
        InB_num_in_4B = 16;
    }
}

vector<vector<int32_t>> Prim04::get_output_shape() const {
    auto para = static_pointer_cast<Prim04_Parameter>(_para);
    return {vector<int32_t>{w_wr_real}};
}

void Prim04::execute(const vector<DataBlock> &input,
                     vector<DataBlock> &output) const {

    auto para = static_pointer_cast<Prim04_Parameter>(_para);
    assert(output.size() == 1);

    uint32_t *p_mem_x1 = (uint32_t *)(input[0].get_data().get());
    uint32_t *p_mem_W = (uint32_t *)(input[1].get_data().get());

    Array<int32_t, 1> p_x({length_in_equal});
    for (int32_t i = 0; i < length_in_equal / InA_num_in_4B; ++i) {
        for (int32_t j = 0; j < InA_num_in_4B; ++j) {
            int32_t raw = p_mem_x1[i];
            p_x[InA_num_in_4B * i + j] = transType(raw, para->x1_precision, j);
        }
    }

    int32_t per32cout_in_4B = cout_in_grp / InB_num_in_4B;
    int32_t perGrp_in_4B = para->nif * per32cout_in_4B;

    Array<int32_t, 2> p_InB({para->nif, w_wr_real});

    for (int32_t w_grp_cnt = 0; w_grp_cnt < w_grp_num; ++w_grp_cnt) {
        for (int32_t nif = 0; nif < para->nif; ++nif) {
            for (int32_t i = 0; i < cout_in_grp / InB_num_in_4B; ++i) {
                for (int32_t j = 0; j < InB_num_in_4B; ++j) {
                    int32_t raw = p_mem_W[w_grp_cnt * perGrp_in_4B +
                                          nif * per32cout_in_4B + i];
                    p_InB[nif]
                         [InB_num_in_4B * i + j + w_grp_cnt * cout_in_grp] =
                             transType(raw, para->x2_precision, j);
                }
            }
        }
    }

    Array<int32_t, 1> p_sb({w_wr_real});
    if (para->bias_type == 2 || para->bias_type == 3) {
        uint32_t *sb_mem = (uint32_t *)(input[2].get_data().get());
        for (int i = 0; i < w_wr_real; ++i)
            p_sb[i] = sb_mem[i];
    } else {
        for (int i = 0; i < w_wr_real; ++i)
            p_sb[i] = para->constant_b;
    }

    shared_ptr<int32_t> so_array = new_array<int32_t>(w_wr_real);
    mlp(w_wr_real, para->nif, p_x, p_InB, p_sb, so_array.get());
    output[0].set_data(reinterpret_pointer_cast<uint8_t>(so_array));
}