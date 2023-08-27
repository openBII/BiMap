// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "prim_07.h"
#include "src/simulator/behavior_simulator/util.h"
#include <cmath>
#include <fstream>
#include <iomanip>

#define P07_DEBUG 0

Prim07::Prim07(shared_ptr<Prim07_Parameter> para) : Primitive(SOMA, para) {

    if (para->x1_precision == 0) {
        neuron_num = ceil(double(para->neuron_real_num) / 4.0);
        neuron_real_num_wr = neuron_num * 4;
        x_num_in_4B = 1;
    } else {
        neuron_num = ceil(double(para->neuron_real_num) / 16.0);
        neuron_real_num_wr = neuron_num * 16;
        x_num_in_4B = 4;
    }

    if (para->lut_data_width == 0) {
        LUT_length = 16;
    } else if (para->lut_data_width == 1) {
        LUT_length = 256;
    } else if (para->lut_data_width == 2) {
        LUT_length = 1 << 12;
    } else if (para->lut_data_width == 3) {
        LUT_length = 1 << 16;
    }

    if (para->x2_precision == 0) {
        y_num_in_4B = 1;
    } else {
        y_num_in_4B = 4;
    }

    if (para->x1_precision == 0 && para->x2_precision == 0) {
        Y_neuron_num_wr = ceil(double(para->neuron_real_num) / 4.0) * 4;
    } else {
        Y_neuron_num_wr = ceil(double(para->neuron_real_num) / 16.0) * 16;
    }
}

vector<vector<int32_t>> Prim07::get_output_shape() const {
    auto para = static_pointer_cast<Prim07_Parameter>(_para);
    return {vector<int32_t>{para->group_num, Y_neuron_num_wr / y_num_in_4B}};
}

void Prim07::execute(const vector<DataBlock> &input,
                     vector<DataBlock> &output) const {

    auto para = static_pointer_cast<Prim07_Parameter>(_para);
    assert(output.size() == 1);

    uint32_t *p_mem_in = (uint32_t *)(input[0].get_data().get());
    uint32_t *p_mem_lut = (uint32_t *)(input[1].get_data().get());

    auto x_array = new_array<int32_t>(para->group_num, neuron_real_num_wr);
    int32_t **array_x = (int32_t **)x_array.get();

    int32_t pergrp_in_4B = neuron_num * 4;

    if (para->x1_precision == 0) {
        for (int32_t k = 0; k < para->group_num; ++k) {
            for (int32_t i = 0; i < neuron_real_num_wr / x_num_in_4B; ++i) {
                for (int32_t j = 0; j < x_num_in_4B; ++j) {
                    array_x[k][x_num_in_4B * i + j] =
                        p_mem_in[int32_t(k * pergrp_in_4B + i)];
                }
            }
        }
    } else {
        for (int32_t k = 0; k < para->group_num; ++k) {
            for (int32_t i = 0; i < neuron_real_num_wr / x_num_in_4B; ++i) {
                for (int32_t j = 0; j < x_num_in_4B; ++j) {
                    array_x[k][x_num_in_4B * i + j] = int8_t(
                        p_mem_in[int32_t(k * pergrp_in_4B + i)] >> (j * 8) &
                        0xff);
                }
            }
        }
    }

    auto lut_array = new_array<int32_t>(LUT_length);
    int32_t *array_lut = lut_array.get();

    if (para->x2_precision == 0) {
        for (int32_t i = 0; i < LUT_length; ++i) {
            array_lut[i] = p_mem_lut[i];
        }
    } else {
        for (int32_t i = 0; i < LUT_length / y_num_in_4B; ++i) {
            for (int32_t j = 0; j < y_num_in_4B; ++j) {
                array_lut[y_num_in_4B * i + j] =
                    int8_t(p_mem_lut[i] >> (8 * j) & 0xff);
            }
        }
    }

    shared_ptr<int32_t> so_array = execute(array_x, array_lut);

    output[0].set_data(reinterpret_pointer_cast<uint8_t>(so_array));
}

shared_ptr<int32_t> Prim07::execute(int32_t **array_in,
                                    int32_t *array_lut) const {

    auto para = static_pointer_cast<Prim07_Parameter>(_para);
    shared_ptr<int32_t> p_convert_addr =
        new_array<int32_t>(para->group_num, neuron_real_num_wr);

    int32_t **convert_addr = (int32_t **)p_convert_addr.get();

    int32_t divider = (pow(2, para->bit_shift_num * 2));

    if (para->lut_data_width == 3) {
        if (para->x1_precision == 0) {
            for (int32_t i = 0; i < para->group_num; ++i) {
                for (int32_t j = 0; j < neuron_real_num_wr; ++j) {
                    convert_addr[i][j] =
                        floor(double(array_in[i][j]) / divider);
                    if (convert_addr[i][j] > 0x7fff) {
                        convert_addr[i][j] = 0x7fff;
                    } else if (convert_addr[i][j] < -32768) {
                        convert_addr[i][j] = -32768;
                    }
                }
            }
        }
    } else if (para->lut_data_width == 2) {
        if (para->x1_precision == 0) {
            for (int32_t i = 0; i < para->group_num; ++i) {
                for (int32_t j = 0; j < neuron_real_num_wr; ++j) {
                    convert_addr[i][j] =
                        floor(double(array_in[i][j]) / divider);
                    if (convert_addr[i][j] > 2047) {
                        convert_addr[i][j] = 2047;
                    } else if (convert_addr[i][j] < -2048) {
                        convert_addr[i][j] = -2048;
                    }
                }
            }
        }
    } else if (para->lut_data_width == 1) {
        if (para->x1_precision == 0) {
            for (int32_t i = 0; i < para->group_num; ++i) {
                for (int32_t j = 0; j < neuron_real_num_wr; ++j) {
                    convert_addr[i][j] =
                        floor(double(array_in[i][j]) / divider);
                    if (convert_addr[i][j] > 127) {
                        convert_addr[i][j] = 127;
                    } else if (convert_addr[i][j] < -128) {
                        convert_addr[i][j] = -128;
                    }
                }
            }
        } else {
            convert_addr = array_in;
        }
    } else if (para->lut_data_width == 0) {
        for (int32_t i = 0; i < para->group_num; ++i) {
            for (int32_t j = 0; j < neuron_real_num_wr; ++j) {
                convert_addr[i][j] = floor(double(array_in[i][j]) / divider);
                if (convert_addr[i][j] > 7) {
                    convert_addr[i][j] = 7;
                } else if (convert_addr[i][j] < -8) {
                    convert_addr[i][j] = -8;
                }
            }
        }
    }

#if P07_DEBUG

    fstream out_x1("convert_cut_addr.txt", ios::out);
    for (int32_t cnt = 0; cnt < para->group_num; ++cnt) {
        for (int32_t i = 0; i < neuron_real_num_wr; ++i) {
            out_x1 << convert_addr[cnt][i] << endl;
        }
    }
    out_x1.close();

    fstream out_x2("convert_lut.txt", ios::out);
    for (int32_t cnt = 0; cnt < LUT_length; ++cnt) {
        out_x2 << array_lut[cnt] << endl;
    }
    out_x2.close();

#endif
    shared_ptr<int32_t> so_array =
        new_array<int32_t>(para->group_num, Y_neuron_num_wr);

    int32_t **array_out = (int32_t **)so_array.get();

    for (int32_t i = 0; i < para->group_num; ++i) {
        for (int32_t j = 0; j < neuron_real_num_wr; ++j) {
            array_out[i][j] =
                array_lut[convert_addr[i][j] >= 0
                              ? convert_addr[i][j]
                              : (LUT_length + convert_addr[i][j])];
        }
    }

    shared_ptr<int32_t> new_so_array =
        new_array<int32_t>(para->group_num * Y_neuron_num_wr / y_num_in_4B);

    int32_t *new_array_out = new_so_array.get();

    if (para->x2_precision == 0) {
        for (int32_t i = 0; i < para->group_num; ++i) {
            for (int32_t j = 0; j < Y_neuron_num_wr / y_num_in_4B; ++j) {
                new_array_out[i * Y_neuron_num_wr / y_num_in_4B + j] =
                    array_out[i][j];
            }
        }
    } else {
        for (int32_t i = 0; i < para->group_num; ++i) {
            for (int32_t j = 0; j < Y_neuron_num_wr / y_num_in_4B; ++j) {
                for (int32_t k = 0; k < y_num_in_4B; ++k) {
                    new_array_out[i * Y_neuron_num_wr / y_num_in_4B + j] |=
                        (array_out[i][j * y_num_in_4B + k] & 0xff) << (8 * k);
                }
            }
        }
    }

#if P07_DEBUG
    fstream out_o("output_raw.txt", ios::out);
    for (int32_t i = 0; i < para->group_num; ++i) {
        for (int32_t j = 0; j < Y_neuron_num_wr; ++j) {
            out_o << array_out[i][j] << endl;
        }
    }
    out_o.close();

#endif
    return new_so_array;
}
