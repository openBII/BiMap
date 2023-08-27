// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "prim_03.h"
#include "src/simulator/behavior_simulator/util.h"
#include <cmath>
#include <fstream>
#include <iomanip>

#define P03_DEBUG 0

Prim03::Prim03(shared_ptr<Prim03_Parameter> para) : Primitive(AXON, para) {
    cin_wr_real = ceil(double(para->nif) / 32.0) * 32;
    if (para->tensor_en) {
        Oy = ((para->ny - 1) / para->stride_y) + 1;
        Ox = ((para->nx - 1) / para->stride_x) + 1;
    }
    num_in_4B = 4;
}

vector<vector<int32_t>> Prim03::get_output_shape() const {
    auto para = static_pointer_cast<Prim03_Parameter>(_para);
    if (para->tensor_en) {
        return {vector<int32_t>{Oy, Ox, cin_wr_real}};
    } else {
        return {vector<int32_t>{para->n_branch, cin_wr_real}};
    }
}

void Prim03::execute(const vector<DataBlock> &input,
                     vector<DataBlock> &output) const {

    auto para = static_pointer_cast<Prim03_Parameter>(_para);
    assert(output.size() == 1);

    uint32_t *p_mem_x1 = (uint32_t *)(input[0].get_data().get());
    uint32_t *p_mem_x2 = (uint32_t *)(input[1].get_data().get());

    int32_t perPx_in_4B = ceil(double(para->nif) / 32.0) *
                          (32 / 4);  // 每个点的所有cin占多少的4bytes
    int32_t perPy_in_4B =
        para->nx * perPx_in_4B;  // 输入图像的每一行的所有cin占多少的4bytes
    if (para->tensor_en) {
        Array<int32_t, 3> p_x1({para->ny, para->nx, cin_wr_real});
        Array<int32_t, 3> p_x2({para->ny, para->nx, cin_wr_real});
        for (int32_t py = 0; py < para->ny; ++py) {
            for (int32_t px = 0; px < para->nx; ++px) {
                for (int32_t i = 0; i < (cin_wr_real / num_in_4B); ++i) {
                    for (int32_t j = 0; j < num_in_4B; ++j) {
                        int32_t temp =
                            p_mem_x1[py * perPy_in_4B + px * perPx_in_4B + i];
                        transType(temp, para->x1_precision, j);
                        p_x1[py][px][num_in_4B * i + j] = temp;

                        temp =
                            p_mem_x2[py * perPy_in_4B + px * perPx_in_4B + i];
                        transType(temp, para->x1_precision, j);
                        p_x2[py][px][num_in_4B * i + j] = temp;
                    }
                }
            }
        }
        shared_ptr<int32_t> sb_array;
        int32_t *p_sb = nullptr;
        if (para->bias_type == 2 || para->bias_type == 3) {
            p_sb = (int32_t *)(input[2].get_data().get());
        }
        shared_ptr<int32_t> so_array = execute(p_x1, p_x2, p_sb);
        output[0].set_data(reinterpret_pointer_cast<uint8_t>(so_array));
    } else {
        Array<int32_t, 2> p_x1({para->n_branch, cin_wr_real});
        Array<int32_t, 2> p_x2({para->n_branch, cin_wr_real});
        for (int32_t px = 0; px < para->n_branch; ++px) {
            for (int32_t i = 0; i < (cin_wr_real / num_in_4B); ++i) {
                for (int32_t j = 0; j < num_in_4B; ++j) {
                    int32_t temp = p_mem_x1[px * perPx_in_4B + i];
                    transType(temp, para->x1_precision, j);
                    p_x1[px][num_in_4B * i + j] = temp;

                    temp = p_mem_x2[px * perPx_in_4B + i];
                    transType(temp, para->x1_precision, j);
                    p_x2[px][num_in_4B * i + j] = temp;
                }
            }
        }
        shared_ptr<int32_t> sb_array;
        int32_t *p_sb = nullptr;
        if (para->bias_type == 2 || para->bias_type == 3) {
            p_sb = (int32_t *)(input[2].get_data().get());
        }
        shared_ptr<int32_t> so_array = execute(p_x1, p_x2, p_sb);
        output[0].set_data(reinterpret_pointer_cast<uint8_t>(so_array));
    }
}

shared_ptr<int32_t> Prim03::execute(Array<int32_t, 3> &p_x1,
                                    Array<int32_t, 3> &p_x2,
                                    int32_t *SB) const {
    auto para = static_pointer_cast<Prim03_Parameter>(_para);
    auto new_so_array = new_array<int32_t>(Oy * Ox * cin_wr_real);
    int32_t *presult = new_so_array.get();

    for (int32_t oy = 0; oy < Oy; oy++) {
        for (int32_t ox = 0; ox < Ox; ox++) {
            for (int32_t f = 0; f < cin_wr_real; f++) {
                int64_t result =
                    ((para->bias_type == 2 || para->bias_type == 3))
                        ? SB[f]
                        : para->constant_b;
                result +=
                    int64_t(p_x1[oy * para->stride_y][ox * para->stride_x][f]) *
                    int64_t(p_x2[oy * para->stride_y][ox * para->stride_x][f]);
                presult[oy * Ox * cin_wr_real + ox * cin_wr_real + f] =
                    sign_cast_64_32(result);
            }
        }
    }
    return new_so_array;
}
shared_ptr<int32_t> Prim03::execute(Array<int32_t, 2> &p_x1,
                                    Array<int32_t, 2> &p_x2,
                                    int32_t *SB) const {
    auto para = static_pointer_cast<Prim03_Parameter>(_para);
    auto new_so_array = new_array<int32_t>(para->n_branch * cin_wr_real);
    int32_t *presult = new_so_array.get();

    for (int32_t ox = 0; ox < para->n_branch; ox++) {
        for (int32_t f = 0; f < cin_wr_real; f++) {
            int64_t result = ((para->bias_type == 2 || para->bias_type == 3))
                                 ? SB[f]
                                 : para->constant_b;
            result += int64_t(p_x1[ox][f]) * int64_t(p_x2[ox][f]);
            presult[ox * cin_wr_real + f] = sign_cast_64_32(result);
        }
    }
    return new_so_array;
}
