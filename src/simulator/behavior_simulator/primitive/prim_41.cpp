// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "prim_41.h"
#include "src/simulator/behavior_simulator/util.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

Prim41::Prim41(shared_ptr<Prim41_Parameter> para) : Primitive(AXON, para) {
    if (para->x1_precision == 3) {
        Km_num = ceil(double(para->nif) / 64.0);
        cin_wr_real = Km_num * 64;
    } else {
        Km_num = ceil(double(para->nif) / 16.0);
        cin_wr_real = Km_num * 16;
    }
    Output_fm_Ox = int32_t(para->nix + para->pad_left + para->pad_right -
                           ((para->nkx - 1) * para->dilate_x + 1)) /
                       para->stride_x +
                   1;
    Output_fm_Oy = int32_t(para->niy + para->pad_top + para->pad_down -
                           ((para->nky - 1) * para->dilate_y + 1)) /
                       para->stride_y +
                   1;

    w_grp_num = ceil(double(para->nof) / 32.0);
    cout_wr_real = w_grp_num * 32;
    // para->nof = cout_wr_real;
}

vector<vector<int32_t>> Prim41::get_output_shape() const {
    auto para = static_pointer_cast<Prim41_Parameter>(_para);
    return {vector<int32_t>{Output_fm_Oy, Output_fm_Ox, cout_wr_real}};
}

void Prim41::mem2si(uint32_t *p_mem_si, Array<int32_t, 3> &p_si) const {
    auto para = static_pointer_cast<Prim41_Parameter>(_para);
    int i = 0;
    for (int y = 0; y < para->niy; ++y)
        for (int x = 0; x < para->nix; ++x)
            for (int f = 0; f < cin_wr_real; ++i) {
                int raw = p_mem_si[i];
                switch (para->x1_precision) {
                    case 0:  // int32
                        p_si[y][x][f++] = raw;
                        break;
                    case 1:  // int8
                        for (int j = 0; j < 4; ++j)
                            p_si[y][x][f++] =
                                (int8_t)((raw >> (j << 3)) & 0xff);
                        break;
                    case 2:
                        for (int j = 0; j < 4; ++j)
                            p_si[y][x][f++] =
                                (uint8_t)((raw >> (j << 3)) & 0xff);
                        break;
                    case 3:  // Ternary
                        for (int j = 0; j < 16; ++j)
                            p_si[y][x][f++] = (int2_t)((raw >> (j << 1)) & 0x3);
                        break;
                    default:
                        break;
                }
            }
}
void Prim41::mem2sw(uint32_t *p_mem_sw, Array<int32_t, 4> &p_sw) const {
    // weight
    auto para = static_pointer_cast<Prim41_Parameter>(_para);
    int32_t per32cout_in_bytes = para->x2_precision == 3 ? 8 : 32;
    int32_t per32cout_in_4B = per32cout_in_bytes / 4;
    int32_t num_in_4B = para->x2_precision == 3 ? 16 : 4;

    int32_t perKx_in_4B =
        para->nif * per32cout_in_4B;  // kernel中一个点的每32个weight占的4B数
    int32_t perKy_in_4B =
        para->nkx *
        perKx_in_4B;  // kernel中的一整行的所有点的32个weight占的4B数
    int32_t perGrp_in_4B =
        para->nky * perKy_in_4B;  // 每个grp的所有weight占的4B数

    for (int32_t w_grp_cnt = 0; w_grp_cnt < w_grp_num; ++w_grp_cnt)
        for (int32_t ky = 0; ky < para->nky; ++ky)
            for (int32_t kx = 0; kx < para->nkx; ++kx)
                for (int32_t nif = 0; nif < para->nif; ++nif)
                    for (int32_t i = 0; i < (32 / num_in_4B); ++i)
                        for (int32_t j = 0; j < num_in_4B; ++j) {
                            int32_t raw = p_mem_sw[int32_t(
                                w_grp_cnt * perGrp_in_4B + ky * perKy_in_4B +
                                kx * perKx_in_4B + nif * per32cout_in_4B + i)];
                            p_sw[num_in_4B * i + j + 32 * w_grp_cnt][ky][kx]
                                [nif] = transType(raw, para->x2_precision, j);
                        }
}
void Prim41::execute(const vector<DataBlock> &input,
                     vector<DataBlock> &output) const {
    auto para = static_pointer_cast<Prim41_Parameter>(_para);
    assert(output.size() == 1);

    uint32_t *p_mem_si = (uint32_t *)(input[0].get_data().get());
    uint32_t *p_mem_sw = (uint32_t *)(input[1].get_data().get());
    Array<int32_t, 3> p_si({para->niy, para->nix, cin_wr_real});
    Array<int32_t, 4> p_sw({cout_wr_real, para->nky, para->nkx, para->nif});
    mem2si(p_mem_si, p_si);
    mem2sw(p_mem_sw, p_sw);
    // load bias vector
    Array<int32_t, 1> p_sb({para->nof});
    if (para->bias_type == 2 || para->bias_type == 3) {
        uint32_t *sb_mem = (uint32_t *)(input[2].get_data().get());
        for (int i = 0; i < para->nof; ++i)
            p_sb[i] = sb_mem[i];
    } else {
        for (int i = 0; i < para->nof; ++i)
            p_sb[i] = 0;
    }

    pad(p_si);
    auto so_array =
        new_array<int32_t>(Output_fm_Oy * Output_fm_Ox * cout_wr_real);
    conv(Output_fm_Oy, Output_fm_Ox, para->nof, para->nif, para->nky, para->nkx,
         para->stride_y, para->stride_x, para->dilate_y, para->dilate_x, p_si,
         p_sw, p_sb.raw(), so_array.get(), Output_fm_Ox * cout_wr_real,
         cout_wr_real);

    output[0].set_data(reinterpret_pointer_cast<uint8_t>(so_array));
}

void Prim41::pad(Array<int32_t, 3> &SI) const {
    auto para = static_pointer_cast<Prim41_Parameter>(_para);
    int32_t Py_real =
        para->niy + para->pad_top +
        para->pad_down;  // vertical length of the input after padding
    int32_t Px_real =
        para->nix + para->pad_left +
        para->pad_right;  // horizontal length of the input after padding
    if (Py_real != para->niy || Px_real != para->nix) {
        Array<int32_t, 3> SI_padding({Py_real, Px_real, cin_wr_real});
        for (int32_t NOY_cnt = (para->pad_top - 1);
             NOY_cnt < (para->pad_top + para->niy - 1); ++NOY_cnt) {
            for (int32_t NOX_cnt = (para->pad_left - 1);
                 NOX_cnt < (para->pad_left + para->nix - 1); ++NOX_cnt) {
                for (int32_t CIN_cnt = 0; CIN_cnt < cin_wr_real; ++CIN_cnt) {
                    SI_padding[NOY_cnt + 1][NOX_cnt + 1][CIN_cnt] =
                        SI[NOY_cnt + 1 - para->pad_top]
                          [NOX_cnt + 1 - para->pad_left][CIN_cnt];
                }
            }
        }
        SI = move(SI_padding);
    }
}
