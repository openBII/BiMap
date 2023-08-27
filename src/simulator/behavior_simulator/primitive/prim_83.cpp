// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "prim_83.h"
#include <cmath>
#include <fstream>

#include "src/simulator/behavior_simulator/util.h"
#include <iomanip>
#include <iostream>
#define P83_DEBUG 0
using namespace std;
Prim83::Prim83(shared_ptr<Prim83_Parameter> para) : Primitive(AXON, para) {

    Km_num = ceil(double(para->nif) / 32.0);
    cin_wr_real = Km_num * 32;
    num_in_4B = 4;

    if (para->tensor_en) {
        Oy = int32_t((para->ny - 1) / para->stride_y) + 1;
        Ox = int32_t((para->nx - 1) / para->stride_x) + 1;
    }
}

vector<vector<int32_t>> Prim83::get_output_shape() const {
    auto para = static_pointer_cast<Prim83_Parameter>(_para);
    if (para->tensor_en) {
        return {vector<int32_t>{Oy, Ox, cin_wr_real}};
    } else {
        return {vector<int32_t>{para->n_branch, cin_wr_real}};
    }
}

void Prim83::execute(const vector<DataBlock> &input,
                     vector<DataBlock> &output) const {

    assert(output.size() == 1);
    auto para = static_pointer_cast<Prim83_Parameter>(_para);

    uint32_t *p_mem_x1 = (uint32_t *)(input[0].get_data().get());

    int32_t perPx_in_4B = Km_num * (32 / 4);  // 每个点的所有cin占多少的4bytes
    int32_t perPy_in_4B =
        para->nx * perPx_in_4B;  // 输入图像的每一行的所有cin占多少的4bytes

    if (para->tensor_en) {
        Array<int32_t, 3> p_x1({para->ny, para->nx, cin_wr_real});
        for (int32_t py = 0; py < para->ny; ++py) {
            for (int32_t px = 0; px < para->nx; ++px) {
                for (int32_t i = 0; i < (cin_wr_real / num_in_4B); ++i) {
                    for (int32_t j = 0; j < num_in_4B; ++j) {
                        auto temp = p_mem_x1[int32_t(py * perPy_in_4B +
                                                     px * perPx_in_4B + i)];
                        switch (para->x1_precision) {
                            case 1:
                                temp >>= (j * 8) & 0xff;
                                temp = int8_t(temp);
                                break;
                            case 2:
                                temp >>= (j * 8) & 0xff;
                                temp = uint8_t(temp);
                                break;
                                //    case 3: temp >>=   (j * 2) & 0b11 ; temp =
                                //    int2_t(temp); break;
                            default:
                                break;
                        }
                        p_x1[py][px][num_in_4B * i + j] = temp;
                    }
                }
            }
        }
        shared_ptr<int32_t> sb_array;
        int32_t *p_sb = nullptr;
        if (para->bias_type == 2 || para->bias_type == 3) {
            p_sb = (int32_t *)(input[1].get_data().get());
        }
        shared_ptr<int32_t> so_array = execute(p_x1, p_sb);

        output[0].set_data(reinterpret_pointer_cast<uint8_t>(so_array));
    } else {
        Array<int32_t, 2> p_x1({para->n_branch, cin_wr_real});
        for (int32_t px = 0; px < para->n_branch; ++px) {
            for (int32_t i = 0; i < (cin_wr_real / num_in_4B); ++i) {
                for (int32_t j = 0; j < num_in_4B; ++j) {
                    auto temp = p_mem_x1[int32_t(px * perPx_in_4B + i)];
                    temp >>= (j * 8) & 0xff;
                    switch (para->x1_precision) {
                        case 1:
                            temp = int8_t(temp);
                            break;
                        case 2:
                            temp = uint8_t(temp);
                            break;
                        default:
                            break;
                    }
                    p_x1[px][num_in_4B * i + j] = temp;
                }
            }
        }
        shared_ptr<int32_t> sb_array;
        int32_t *p_sb = nullptr;
        if (para->bias_type == 2 || para->bias_type == 3) {
            p_sb = (int32_t *)(input[1].get_data().get());
        }
        shared_ptr<int32_t> so_array = execute(p_x1, p_sb);
        output[0].set_data(reinterpret_pointer_cast<uint8_t>(so_array));
    }
}

shared_ptr<int32_t> Prim83::execute(Array<int32_t, 3> &p_x1,
                                    int32_t *SB) const {

    auto para = static_pointer_cast<Prim83_Parameter>(_para);
    auto new_so_array = new_array<int32_t>(Oy * Ox * cin_wr_real);
    int32_t *presult = new_so_array.get();
    for (int32_t oy = 0; oy < Oy; oy++) {
        for (int32_t ox = 0; ox < Ox; ox++) {
            for (int32_t f = 0; f < cin_wr_real; f++) {
                int64_t result = (para->bias_type == 0 || para->bias_type == 1)
                                     ? para->constant_b
                                     : SB[f];
                result +=
                    int64_t(p_x1[oy * para->stride_y][ox * para->stride_x][f]) *
                    int64_t(para->constant_a);
                presult[oy * Ox * cin_wr_real + ox * cin_wr_real + f] =
                    sign_cast_64_32(result);
            }
        }
    }

    return new_so_array;
}
shared_ptr<int32_t> Prim83::execute(Array<int32_t, 2> &p_x1,
                                    int32_t *SB) const {
    auto para = static_pointer_cast<Prim83_Parameter>(_para);
    auto new_so_array = new_array<int32_t>(para->n_branch * cin_wr_real);
    int32_t *presult = new_so_array.get();
    for (int32_t ox = 0; ox < para->n_branch; ox++) {
        for (int32_t f = 0; f < cin_wr_real; f++) {
            int64_t result = (para->bias_type == 0 || para->bias_type == 1)
                                 ? para->constant_b
                                 : SB[f];
            result += int64_t(p_x1[ox][f]) * int64_t(para->constant_a);
            presult[ox * cin_wr_real + f] = sign_cast_64_32(result);
        }
    }
    return new_so_array;
}