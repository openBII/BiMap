// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "prim_02.h"
#include "src/simulator/behavior_simulator/patch.h"
#include "src/simulator/behavior_simulator/util.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#define P02_DEBUG 0
void Prim02::pad(Array<int32_t, 3> &p_si) const {
    auto para = static_pointer_cast<Prim02_Parameter>(_para);

    if (!para->has_pad)
        return;

    if ((Px_real != para->nix || para->niy != Py_real)) {
        Array<int32_t, 3> SI_padding({Py_real, Px_real, cin_wr_real});
        for (int32_t NOY_cnt = para->pad_top;
             NOY_cnt < para->pad_top + para->niy; ++NOY_cnt) {
            for (int32_t NOX_cnt = para->pad_left;
                 NOX_cnt < para->pad_left + para->nix; ++NOX_cnt) {
                for (int32_t CIN_cnt = 0; CIN_cnt < cin_wr_real; ++CIN_cnt) {
                    SI_padding[NOY_cnt][NOX_cnt][CIN_cnt] =
                        p_si[NOY_cnt - para->pad_top][NOX_cnt - para->pad_left]
                            [CIN_cnt];
                }
            }
        }

        p_si = move(SI_padding);
    }
}

void pooling(int noy, int nox, int nf, int nky, int nkx, int nsy, int nsx,
             Array<int32_t, 3> &si, int *b, int *so, int stepy, int stepx) {
    for (int32_t oy = 0; oy < noy; oy++) {
        for (int32_t ox = 0; ox < nox; ox++) {
            for (int32_t f = 0; f < nf; f++) {
                int64_t sum = b[f];
                for (int32_t oky = 0; oky < nky; oky++) {
                    for (int32_t okx = 0; okx < nkx; okx++) {
                        sum += int64_t(si[oky + oy * nsy][okx + ox * nsx][f]);
                        sum = sign_cast_64_32(sum);
                    }
                }
                so[oy * stepy + ox * stepx + f] = sum;
            }
        }
    }
}
// Pooling
Prim02::Prim02(shared_ptr<Prim02_Parameter> para) : Primitive(AXON, para) {
    int precisions[] = {32, 8, 8, 2};
    int precision = precisions[para->x1_precision];

    // nif 切成的 32B 数
    Km_num = ceil(double(para->nif) / (256 / precision));
    cin_wr_real = Km_num * 256 / precision;
    num_in_4B = 32 / precision;

    Px_real = para->nix + para->pad_left + para->pad_right;
    Py_real = para->niy + para->pad_top + para->pad_down;
    array_num = para->nkx * para->nky;
    Output_fm_Ox = int32_t((Px_real - para->nkx) / para->stride_x) + 1;
    Output_fm_Oy = int32_t((Py_real - para->nky) / para->stride_y) + 1;
}

vector<vector<int32_t>> Prim02::get_output_shape() const {
    auto para = static_pointer_cast<Prim02_Parameter>(_para);
    if (para->avg_pooling_en)
        return {{Output_fm_Oy, Output_fm_Ox, cin_wr_real}};
    return {{para->niy, para->nix, cin_wr_real}};
}

void Prim02::mem2si(uint32_t *p_mem_si, Array<int32_t, 4> &p_si) const {
    auto para = static_pointer_cast<Prim02_Parameter>(_para);
    // int32_t perPx_in_4B = Km_num * (32 / 4); //每个点的所有cin占多少的4bytes
    // int32_t perPy_in_4B =
    //     para->nix * perPx_in_4B; // 输入图像的每一行的所有cin占多少的4bytes
    // int32_t perarray_in_4B = para->niy * perPy_in_4B;

    int i = 0;
    for (int a = 0; a < array_num; ++a)
        for (int y = 0; y < para->niy; ++y)
            for (int x = 0; x < para->nix; ++x)
                for (int f = 0; f < cin_wr_real; ++i) {
                    int raw = p_mem_si[i];
                    switch (para->x1_precision) {
                        case 0:  // int32
                            p_si[a][y][x][f++] = raw;
                            break;
                        case 1:  // int8
                            for (int j = 0; j < 4; ++j)
                                p_si[a][y][x][f++] =
                                    (int8_t)((raw >> (j << 3)) & 0xff);
                            break;
                        case 2:
                            for (int j = 0; j < 4; ++j)
                                p_si[a][y][x][f++] =
                                    (uint8_t)((raw >> (j << 3)) & 0xff);
                            break;
                        case 3:  // Ternary
                            for (int j = 0; j < 16; ++j)
                                p_si[a][y][x][f++] =
                                    (int2_t)((raw >> (j << 1)) & 0x3);
                            break;
                        default:
                            break;
                    }
                }

    // for (int32_t P_cnt = 0; P_cnt < array_num; ++P_cnt)
    //     for (int32_t py = 0; py < para->niy; ++py)
    //         for (int32_t px = 0; px < para->nix; ++px)
    //             for (int32_t i = 0; i < cin_wr_real / num_in_4B; ++i)
    //                 for (int32_t j = 0; j < num_in_4B; ++j) {
    //                     int32_t raw = p_mem_si[int32_t(P_cnt * perarray_in_4B
    //                     +
    //                                                    py * perPy_in_4B +
    //                                                    px * perPx_in_4B +
    //                                                    i)];
    //                     p_si[P_cnt][py][px][num_in_4B * i + j] =
    //                         transType(raw, para->x1_precision, j);
    //                 }
}
void Prim02::mem2si(uint32_t *p_mem_si, Array<int32_t, 3> &p_si) const {
    auto para = static_pointer_cast<Prim02_Parameter>(_para);
    int32_t perPx_in_4B = Km_num * (32 / 4);  // 每个点的所有cin占多少的4bytes
    int32_t perPy_in_4B =
        para->nix * perPx_in_4B;  // 输入图像的每一行的所有cin占多少的4bytes
    int32_t perarray_in_4B = para->niy * perPy_in_4B;
    for (int32_t py = 0; py < para->niy; ++py)
        for (int32_t px = 0; px < para->nix; ++px)
            for (int32_t i = 0; i < cin_wr_real / num_in_4B; ++i)
                for (int32_t j = 0; j < num_in_4B; ++j) {
                    int32_t raw = p_mem_si[int32_t(py * perPy_in_4B +
                                                   px * perPx_in_4B + i)];
                    p_si[py][px][num_in_4B * i + j] =
                        transType(raw, para->x1_precision, j);
                }
}

void Prim02::execute(const vector<DataBlock> &input,
                     vector<DataBlock> &output) const {
    auto para = static_pointer_cast<Prim02_Parameter>(_para);
    assert(output.size() == 1);

    uint32_t *p_mem_si = (uint32_t *)(input[0].get_data().get());

    // load bias vector
    Array<int32_t, 1> p_sb({para->nif});
    if (para->bias_type == 2 || para->bias_type == 3) {
        uint32_t *sb_mem = (uint32_t *)(input[1].get_data().get());
        for (int i = 0; i < para->nif; ++i)
            p_sb[i] = sb_mem[i];
    } else {
        for (int i = 0; i < para->nif; ++i)
            p_sb[i] = para->constant_b;
    }
    if (para->avg_pooling_en) {
        Array<int32_t, 3> p_si({para->niy, para->nix, cin_wr_real});
        mem2si(p_mem_si, p_si);
        pad(p_si);
        shared_ptr<int32_t> new_so_array =
            new_array<int32_t>(Output_fm_Oy * Output_fm_Ox * cin_wr_real);
        int32_t *presult = new_so_array.get();
        pooling(Output_fm_Oy, Output_fm_Ox, para->nif, para->nky, para->nkx,
                para->stride_y, para->stride_x, p_si, p_sb.raw(), presult,
                Output_fm_Ox * cin_wr_real, cin_wr_real);

        output[0].set_data(reinterpret_pointer_cast<uint8_t>(new_so_array));
    } else {
        Array<int32_t, 4> p_si({array_num, para->niy, para->nix, cin_wr_real});
        mem2si(p_mem_si, p_si);
        shared_ptr<int32_t> new_so_array =
            new_array<int32_t>(para->niy * para->nix * cin_wr_real);
        int32_t *presult = new_so_array.get();
        vector_add(para->niy, para->nix, para->nif, array_num, p_si, p_sb.raw(),
                   presult, para->nix * cin_wr_real, cin_wr_real);

        output[0].set_data(reinterpret_pointer_cast<uint8_t>(new_so_array));
    }
}
