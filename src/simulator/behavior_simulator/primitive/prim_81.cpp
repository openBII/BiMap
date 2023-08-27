// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "prim_81.h"
#include "src/simulator/behavior_simulator/util.h"
#include <cmath>
#include <fstream>
#include <iomanip>

#define P81_DEBUG 0
Prim81::Prim81(shared_ptr<Prim81_Parameter> para) : Primitive(AXON, para) {

    Px_real = para->nix + para->pad_left + para->pad_right;
    Py_real = para->niy + para->pad_top + para->pad_down;
    Kx_real = (para->nkx - 1) * para->dilate_x + 1;
    Ky_real = (para->nky - 1) * para->dilate_y + 1;
    Output_fm_Ox = int32_t((Px_real - Kx_real) / para->stride_x) + 1;
    Output_fm_Oy = int32_t((Py_real - Ky_real) / para->stride_y) + 1;

    w_grp_num = ceil(double(para->nof) / 32.0);
    cout_wr_real = w_grp_num * 32;

    nix_grp_real = ceil(double(Px_real) / 16.0);
    nix_wr_real = nix_grp_real * 16;
    InA_num_in_4B = 4;

    if (para->x2_precision == 3) {
        InB_num_in_4B = 16;
        InB_per32cout_in_bytes = 8;  // 32个cout的weight占几个1B
    } else {
        InB_num_in_4B = 4;
        InB_per32cout_in_bytes = 32;  // 32个cout的weight占几个1B
    }
    para->nof = cout_wr_real;
}

vector<vector<int32_t>> Prim81::get_output_shape() const {
    auto para = static_pointer_cast<Prim81_Parameter>(_para);
    return {vector<int32_t>{Output_fm_Oy, Output_fm_Ox, cout_wr_real}};
}

void Prim81::execute(const vector<DataBlock> &input,
                     vector<DataBlock> &output) const {

    auto para = static_pointer_cast<Prim81_Parameter>(_para);
    assert(output.size() == 1);

    uint32_t *p_mem_si = (uint32_t *)(input[0].get_data().get());
    uint32_t *p_mem_sw = (uint32_t *)(input[1].get_data().get());

    Array<int32_t, 3> p_si({Py_real, nix_wr_real, para->nif});
    Array<int32_t, 4> p_sw({para->nky, para->nkx, para->nif, cout_wr_real});

    int32_t InA_perPy_in_4B = nix_grp_real * (16 / 4);
    int32_t InA_percin_in_4B = Py_real * InA_perPy_in_4B;

    for (int32_t nif = 0; nif < para->nif; ++nif)
        for (int32_t py = 0; py < Py_real; ++py)
            for (int32_t i = 0; i < (nix_wr_real / InA_num_in_4B); ++i)
                for (int32_t j = 0; j < InA_num_in_4B; ++j) {
                    int32_t raw = p_mem_si[int32_t(py * InA_perPy_in_4B +
                                                   nif * InA_percin_in_4B + i)];
                    p_si[py][InA_num_in_4B * i + j][nif] =
                        transType(raw, para->x1_precision, j);
                }

    // weight

    int32_t InB_per32cout_in_4B =
        InB_per32cout_in_bytes / 4;  // 32个cout的weight占几个4B
    int32_t InB_perKy_in_4B =
        para->nkx *
        InB_per32cout_in_4B;  // kernel中的一整行的所有点的32个weight占的4B数
    int32_t InB_percin_in_4B = para->nky * InB_perKy_in_4B;
    int32_t InB_perGrp_in_4B =
        para->nif * InB_percin_in_4B;  // 每个grp的所有weight占的4B数

    for (int32_t w_grp_cnt = 0; w_grp_cnt < w_grp_num; ++w_grp_cnt)
        for (int32_t nif = 0; nif < para->nif; ++nif)
            for (int32_t ky = 0; ky < para->nky; ++ky)
                for (int32_t kx = 0; kx < para->nkx; ++kx)
                    for (int32_t i = 0; i < (32 / InB_num_in_4B); ++i)
                        for (int32_t j = 0; j < InB_num_in_4B; ++j) {
                            int32_t raw =
                                p_mem_sw[w_grp_cnt * InB_perGrp_in_4B +
                                         nif * InB_percin_in_4B +
                                         ky * InB_perKy_in_4B +
                                         kx * InB_per32cout_in_4B + i];
                            p_sw[ky][kx][nif]
                                [InB_num_in_4B * i + j + 32 * w_grp_cnt] =
                                    transType(raw, para->x2_precision, j);
                        }

    // bias
    Array<int32_t, 1> p_sb({para->nof});
    if (para->bias_type >= 2) {
        uint32_t *sb_mem = (uint32_t *)(input[2].get_data().get());
        for (int i = 0; i < para->nof; ++i)
            p_sb[i] = sb_mem[i];
    } else {
        for (int i = 0; i < para->nof; ++i)
            p_sb[i] = 0;
    }

    auto so_array =
        new_array<int32_t>(Output_fm_Oy * Output_fm_Ox * cout_wr_real);
    conv2d(Output_fm_Oy, Output_fm_Ox, para->nof, para->nif, para->nky,
           para->nkx, para->stride_y, para->stride_x, para->dilate_y,
           para->dilate_x, p_si, p_sw, p_sb.raw(), so_array.get(),
           Output_fm_Ox * cout_wr_real, cout_wr_real);
    output[0].set_data(reinterpret_pointer_cast<uint8_t>(so_array));
}
