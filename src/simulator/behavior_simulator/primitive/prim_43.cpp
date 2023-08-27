// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "prim_43.h"
#include "src/simulator/behavior_simulator/util.h"
#include <cmath>
#include <fstream>
#include <iomanip>

#define P43_DEBUG 0

Prim43::Prim43(shared_ptr<Prim43_Parameter> para) : Primitive(AXON, para) {
    if (para->x1_precision == 2 || para->x1_precision == 1) {
        Km_num = ceil(double(para->x2_length) / 32.0);
        cin_wr_real = Km_num * 32;
        num_in_4B = 4;
    } else {
        Km_num = ceil(double(para->x2_length) / 128.0);
        cin_wr_real = Km_num * 128;
        num_in_4B = 16;
    }
    if (para->tensor_en) {
        Oy = int32_t((para->ny - 1) / para->stride_y) + 1;
        Ox = int32_t((para->nx - 1) / para->stride_x) + 1;
    }
}

vector<vector<int32_t>> Prim43::get_output_shape() const {
    auto para = static_pointer_cast<Prim43_Parameter>(_para);
    if (para->tensor_en) {
        return {vector<int32_t>{Oy, Ox, cin_wr_real}};
    } else {
        return {vector<int32_t>{para->n_branch, cin_wr_real}};
    }
}

void Prim43::execute(const vector<DataBlock> &input,
                     vector<DataBlock> &output) const {

    auto para = static_pointer_cast<Prim43_Parameter>(_para);
    assert(output.size() == 1);

    uint32_t *p_mem_x1 = (uint32_t *)(input[0].get_data().get());
    uint32_t *p_mem_A = (uint32_t *)(input[1].get_data().get());

    int32_t perPx_in_4B = Km_num * (32 / 4);  // 每个点的所有cin占多少的4bytes
    int32_t perPy_in_4B =
        para->nx * perPx_in_4B;  // 输入图像的每一行的所有cin占多少的4bytes

    Array<int32_t, 1> p_InB({para->x2_length});
    for (int32_t i = 0; i < (para->x2_length / 4); ++i) {
        for (int32_t j = 0; j < 4; ++j) {
            int32_t raw = p_mem_A[i];
            p_InB[4 * i + j] = transType(raw, para->x2_precision, j);
        }
    }

    int32_t *p_sb = nullptr;
    if (para->bias_type == 2 || para->bias_type == 3) {
        p_sb = (int32_t *)(input[2].get_data().get());
    }
    if (para->tensor_en) {
        Array<int32_t, 3> p_x1({para->ny, para->nx, cin_wr_real});
        for (int32_t py = 0; py < para->ny; ++py) {
            for (int32_t px = 0; px < para->nx; ++px) {
                for (int32_t i = 0; i < (cin_wr_real / num_in_4B); ++i) {
                    for (int32_t j = 0; j < num_in_4B; ++j) {
                        int32_t raw = p_mem_x1[int32_t(py * perPy_in_4B +
                                                       px * perPx_in_4B + i)];
                        p_x1[py][px][num_in_4B * i + j] =
                            transType(raw, para->x1_precision, j);
                    }
                }
            }
        }
        shared_ptr<int32_t> so_array = execute(p_x1, p_InB, p_sb);

        output[0].set_data(reinterpret_pointer_cast<uint8_t>(so_array));
    } else {
        Array<int32_t, 2> p_x1({para->n_branch, cin_wr_real});
        for (int32_t px = 0; px < para->n_branch; ++px) {
            for (int32_t i = 0; i < (cin_wr_real / num_in_4B); ++i) {
                for (int32_t j = 0; j < num_in_4B; ++j) {
                    int32_t raw = p_mem_x1[int32_t(px * perPx_in_4B + i)];
                    p_x1[px][num_in_4B * i + j] =
                        transType(raw, para->x1_precision, j);
                }
            }
        }
        shared_ptr<int32_t> so_array = execute(p_x1, p_InB, p_sb);
        output[0].set_data(reinterpret_pointer_cast<uint8_t>(so_array));
    }
}

shared_ptr<int32_t> Prim43::execute(Array<int32_t, 3> &p_x1,
                                    Array<int32_t, 1> &p_A, int32_t *SB) const {
    auto para = static_pointer_cast<Prim43_Parameter>(_para);
    bool isBVector = para->bias_type == 2 || para->bias_type == 3;
    shared_ptr<int32_t> new_so_array =
        new_array<int32_t>(Oy * Ox * cin_wr_real);
    int32_t *presult = new_so_array.get();
    for (int32_t oy = 0; oy < Oy; oy++) {
        for (int32_t ox = 0; ox < Ox; ox++) {
            for (int32_t f = 0; f < cin_wr_real; f++) {
                int64_t result = isBVector ? SB[f] : para->constant_b;
                result +=
                    int64_t(p_x1[oy * para->stride_y][ox * para->stride_x][f]) *
                    int64_t(p_A[f]);
                presult[oy * Ox * cin_wr_real + ox * cin_wr_real + f] =
                    sign_cast_64_32(result);
            }
        }
    }
    return new_so_array;
}

shared_ptr<int32_t> Prim43::execute(Array<int32_t, 2> &p_x1,
                                    Array<int32_t, 1> &p_A, int32_t *SB) const {
    auto para = static_pointer_cast<Prim43_Parameter>(_para);
    bool isBVector = para->bias_type == 2 || para->bias_type == 3;
    shared_ptr<int32_t> new_so_array =
        new_array<int32_t>(para->n_branch * cin_wr_real);
    int32_t *presult = new_so_array.get();
    for (int32_t ox = 0; ox < para->n_branch; ox++) {
        for (int32_t f = 0; f < cin_wr_real; f++) {
            int64_t result = isBVector ? SB[f] : para->constant_b;
            result += int64_t(p_x1[ox][f]) * int64_t(p_A[f]);
            presult[ox * cin_wr_real + f] = sign_cast_64_32(result);
        }
    }

    return new_so_array;
}
