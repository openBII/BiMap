// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "prim_25.h"
#include "src/simulator/behavior_simulator/util.h"
#include <cmath>
#include <fstream>
#include <iomanip>

#define P25_DEBUG 0

Prim25::Prim25(shared_ptr<Prim25_Parameter> para) : Primitive(SOMA, para) {

    if (para->x1_precision == 0) {
        Km_num_in = ceil(double(para->nif) / 4.0);
        cin_wr_real = Km_num_in * 4;
        num_in_4B = 1;
    } else if (para->x1_precision == 3) {
        Km_num_in = ceil(double(para->nif) / 64.0);
        cin_wr_real = Km_num_in * 64;
        num_in_4B = 16;
    } else {
        Km_num_in = ceil(double(para->nif) / 16.0);
        cin_wr_real = Km_num_in * 16;
        num_in_4B = 4;
    }
    Px_real = para->nix + para->pad_left + para->pad_right;
    Py_real = para->niy + para->pad_top + para->pad_down;

    Output_fm_Ox = int32_t((Px_real - para->nkx) / para->stride_x) + 1;
    Output_fm_Oy = int32_t((Py_real - para->nky) / para->stride_y) + 1;

    if (para->out_precision == 0) {
        Km_num_out = ceil(double(para->nof) / 4.0);
        cout_real = Km_num_out * 4;
    } else if (para->out_precision == 1) {
        Km_num_out = ceil(double(para->nof) / 16.0);
        cout_real = Km_num_out * 16;
    } else {
        Km_num_out = ceil(double(para->nof) / 64.0);
        cout_real = Km_num_out * 64;
    }
}

vector<vector<int32_t>> Prim25::get_output_shape() const {
    auto para = static_pointer_cast<Prim25_Parameter>(_para);
    if (para->out_precision == 0) {
        return {vector<int32_t>{Output_fm_Oy, Output_fm_Ox, cout_real}};
    } else if (para->out_precision == 1) {
        return {vector<int32_t>{Output_fm_Oy, Output_fm_Ox, cout_real / 4}};
    } else if (para->out_precision == 2) {
        return {vector<int32_t>{Output_fm_Oy, Output_fm_Ox, cout_real / 4}};
    } else {
        return {vector<int32_t>{Output_fm_Oy, Output_fm_Ox, cout_real / 16}};
    }
}

void Prim25::execute(const vector<DataBlock> &input,
                     vector<DataBlock> &output) const {

    auto para = static_pointer_cast<Prim25_Parameter>(_para);
    assert(output.size() == 1);

    uint32_t *p_mem_si = (uint32_t *)(input[0].get_data().get());

    shared_ptr<int32_t> si_array =
        new_array<int32_t>(para->niy, para->nix, cin_wr_real);
    int32_t ***p_si = (int32_t ***)(si_array.get());

    int32_t perPx_in_4B = Km_num_in * (16 / 4);  // 每个点的所有cin占多少的4bytes
    int32_t perPy_in_4B =
        para->nix * perPx_in_4B;  // 输入图像的每一行的所有cin占多少的4bytes

    if (para->x1_precision == 0) {
        for (int32_t py = 0; py < para->niy; ++py) {
            for (int32_t px = 0; px < para->nix; ++px) {
                for (int32_t i = 0; i < cin_wr_real / num_in_4B; ++i) {
                    for (int32_t j = 0; j < num_in_4B; ++j) {
                        p_si[py][px][num_in_4B * i + j] = p_mem_si[int32_t(
                            py * perPy_in_4B + px * perPx_in_4B + i)];
                    }
                }
            }
        }
    } else if (para->x1_precision == 3) {
        for (int32_t py = 0; py < para->niy; ++py) {
            for (int32_t px = 0; px < para->nix; ++px) {
                for (int32_t i = 0; i < cin_wr_real / num_in_4B; ++i) {
                    for (int32_t j = 0; j < num_in_4B; ++j) {
                        auto temp = p_mem_si[int32_t(py * perPy_in_4B +
                                                     px * perPx_in_4B + i)] >>
                                        (j * 2) &
                                    0b11;
                        p_si[py][px][num_in_4B * i + j] = temp == 3 ? -1 : temp;
                    }
                }
            }
        }
    } else if (para->x1_precision == 2) {
        for (int32_t py = 0; py < para->niy; ++py) {
            for (int32_t px = 0; px < para->nix; ++px) {
                for (int32_t i = 0; i < cin_wr_real / num_in_4B; ++i) {
                    for (int32_t j = 0; j < num_in_4B; ++j) {
                        p_si[py][px][num_in_4B * i + j] =
                            (uint8_t)(p_mem_si[int32_t(py * perPy_in_4B +
                                                       px * perPx_in_4B + i)] >>
                                          (j * 8) &
                                      0xff);
                    }
                }
            }
        }
    } else if (para->x1_precision == 1) {
        for (int32_t py = 0; py < para->niy; ++py) {
            for (int32_t px = 0; px < para->nix; ++px) {
                for (int32_t i = 0; i < cin_wr_real / num_in_4B; ++i) {
                    for (int32_t j = 0; j < num_in_4B; ++j) {
                        p_si[py][px][num_in_4B * i + j] =
                            (int8_t)(p_mem_si[int32_t(py * perPy_in_4B +
                                                      px * perPx_in_4B + i)] >>
                                         (j * 8) &
                                     0xff);
                    }
                }
            }
        }
    }

    shared_ptr<int32_t> so_array = execute(p_si);

    output[0].set_data(reinterpret_pointer_cast<uint8_t>(so_array));
}

shared_ptr<int32_t> Prim25::execute(int32_t ***SI) const {

    auto para = static_pointer_cast<Prim25_Parameter>(_para);

    shared_ptr<int32_t> new_so_array = nullptr;

    shared_ptr<int32_t> pad_array = nullptr;
    int32_t ***SI_padding = SI;
    if ((Px_real != para->nix || para->niy != Py_real)) {
        pad_array = new_array<int32_t>(Py_real, Px_real, cin_wr_real);
        SI_padding = (int32_t ***)(pad_array.get());

        for (int32_t NOY_cnt = (para->pad_top - 1);
             NOY_cnt < (para->pad_top + para->niy - 1); ++NOY_cnt) {
            for (int32_t NOX_cnt = (para->pad_left - 1);
                 NOX_cnt < (para->pad_left + para->nix - 1); ++NOX_cnt) {
                for (int32_t CIN_cnt = 0; CIN_cnt < cin_wr_real; ++CIN_cnt) {
                    SI_padding[NOY_cnt + 1][NOX_cnt + 1][CIN_cnt] =
                        (SI)[NOY_cnt + 1 - para->pad_top]
                            [NOX_cnt + 1 - para->pad_left][CIN_cnt];
                }
            }
        }
    }

    double cut_num = pow(2, para->bit_shift_num * 2);

    auto pX_array_trans =
        new_array<int32_t>(Py_real, Px_real, max(cin_wr_real, cout_real));
    int32_t ***X_array_trans = (int32_t ***)pX_array_trans.get();

    if (para->x1_precision >= para->out_precision) {
        for (int32_t Py_cnt = 0; Py_cnt < Py_real; ++Py_cnt) {
            for (int32_t Px_cnt = 0; Px_cnt < Px_real; ++Px_cnt) {
                for (int32_t cin_cnt = 0; cin_cnt < cin_wr_real; ++cin_cnt) {
                    X_array_trans[Py_cnt][Px_cnt][cin_cnt] =
                        SI_padding[Py_cnt][Px_cnt][cin_cnt];
                }
            }
        }
    } else {
        int32_t num_max = para->out_precision == 3 ? 1 : 127;
        int32_t num_min = para->out_precision == 3 ? -1 : -128;
        for (int32_t Py_cnt = 0; Py_cnt < Py_real; ++Py_cnt) {
            for (int32_t Px_cnt = 0; Px_cnt < Px_real; ++Px_cnt) {
                for (int32_t cin_cnt = 0; cin_cnt < para->nif; ++cin_cnt) {
                    int32_t temp = floor(
                        double(SI_padding[Py_cnt][Px_cnt][cin_cnt]) / cut_num);
                    if (temp > num_max) {
                        temp = num_max;
                    } else if (temp < num_min) {
                        temp = num_min;
                    }
                    X_array_trans[Py_cnt][Px_cnt][cin_cnt] = temp;
                }
            }
        }
    }

    auto so_array = new_array<int32_t>(Output_fm_Oy, Output_fm_Ox, cout_real);
    int32_t ***result = (int32_t ***)so_array.get();

    for (int32_t f = 0; f < cout_real; f++) {
        for (int32_t oy = 0; oy < Output_fm_Oy; oy++) {
            for (int32_t ox = 0; ox < Output_fm_Ox; ox++) {
                result[oy][ox][f] =
                    X_array_trans[oy * para->stride_y][ox * para->stride_x][f];
                for (int32_t oky = 0; oky < para->nky; oky++) {
                    for (int32_t okx = 0; okx < para->nkx; okx++) {
                        result[oy][ox][f] =
                            min(result[oy][ox][f],
                                X_array_trans[oky + oy * para->stride_y]
                                             [okx + ox * para->stride_x][f]);
                    }
                }
            }
        }
    }

    if ((para->x1_precision == 0 && para->out_precision == 1) ||
        (para->x1_precision == 1 && para->out_precision == 1)) {
        for (int32_t f = 0; f < cout_real / 4; f++) {
            for (int32_t cnt = 0; cnt < 4; cnt++) {
                int32_t CMP_C_real = para->compare_init >> (cnt * 8) & 0xff;
                CMP_C_real =
                    CMP_C_real > 0x7f ? CMP_C_real - 0x100 : CMP_C_real;
                for (int32_t oy = 0; oy < Output_fm_Oy; oy++) {
                    for (int32_t ox = 0; ox < Output_fm_Ox; ox++) {
                        result[oy][ox][4 * f + cnt] =
                            min(CMP_C_real, result[oy][ox][4 * f + cnt]);
                    }
                }
            }
        }
    } else if ((para->x1_precision == 0 && para->out_precision == 3) ||
               (para->x1_precision == 1 && para->out_precision == 3) ||
               (para->x1_precision == 3 && para->out_precision == 3)) {
        for (int32_t f = 0; f < cout_real / 16; f++) {
            for (int32_t cnt = 0; cnt < 16; cnt++) {
                int32_t CMP_C_real = para->compare_init >> (cnt * 2) & 0b11;
                CMP_C_real = CMP_C_real > 1 ? -1 : CMP_C_real;
                for (int32_t oy = 0; oy < Output_fm_Oy; oy++) {
                    for (int32_t ox = 0; ox < Output_fm_Ox; ox++) {
                        result[oy][ox][16 * f + cnt] =
                            min(CMP_C_real, result[oy][ox][16 * f + cnt]);
                    }
                }
            }
        }
    } else if (para->x1_precision == 0 && para->out_precision == 0) {
        for (int32_t f = 0; f < cout_real; f++) {
            for (int32_t oy = 0; oy < Output_fm_Oy; oy++) {
                for (int32_t ox = 0; ox < Output_fm_Ox; ox++) {

                    result[oy][ox][f] =
                        min(para->compare_init, result[oy][ox][f]);
                }
            }
        }
    } else if (para->x1_precision == 3 && para->out_precision == 1) {
        for (int32_t f = 0; f < cout_real / 16; f++) {
            for (int32_t cnt = 0; cnt < 16; cnt++) {
                int32_t CMP_C_real = para->compare_init >> (cnt * 2) & 0b11;
                CMP_C_real = CMP_C_real > 1 ? -1 : CMP_C_real;
                for (int32_t oy = 0; oy < Output_fm_Oy; oy++) {
                    for (int32_t ox = 0; ox < Output_fm_Ox; ox++) {
                        result[oy][ox][16 * f + cnt] =
                            min(CMP_C_real, result[oy][ox][16 * f + cnt]);
                    }
                }
            }
        }
    }

    int32_t *pnew_result = nullptr;

    if (para->out_precision == 0) {
        new_so_array =
            new_array<int32_t>(Output_fm_Oy * Output_fm_Ox * cout_real);
        pnew_result = new_so_array.get();
        for (int32_t oy = 0; oy < Output_fm_Oy; oy++) {
            for (int32_t ox = 0; ox < Output_fm_Ox; ox++) {
                for (int32_t f = 0; f < cout_real; f++) {
                    pnew_result[oy * Output_fm_Ox * cout_real + ox * cout_real +
                                f] = result[oy][ox][f];
                }
            }
        }
    } else if (para->out_precision == 1 || para->out_precision == 2) {
        new_so_array =
            new_array<int32_t>(Output_fm_Oy * Output_fm_Ox * cout_real / 4);
        pnew_result = new_so_array.get();
        for (int32_t oy = 0; oy < Output_fm_Oy; oy++) {
            for (int32_t ox = 0; ox < Output_fm_Ox; ox++) {
                for (int32_t f = 0; f < cout_real / 4; f++) {
                    for (int32_t cnt = 0; cnt < 4; ++cnt) {
                        pnew_result[oy * Output_fm_Ox * cout_real / 4 +
                                    ox * cout_real / 4 + f] |=
                            (result[oy][ox][f * 4 + cnt] & 0xff) << (cnt * 8);
                    }
                }
            }
        }
    } else {
        new_so_array =
            new_array<int32_t>(Output_fm_Oy * Output_fm_Ox * cout_real / 16);
        pnew_result = new_so_array.get();
        for (int32_t oy = 0; oy < Output_fm_Oy; oy++) {
            for (int32_t ox = 0; ox < Output_fm_Ox; ox++) {
                for (int32_t f = 0; f < cout_real / 16; f++) {
                    for (int32_t cnt = 0; cnt < 16; ++cnt) {
                        pnew_result[oy * Output_fm_Ox * cout_real / 16 +
                                    ox * cout_real / 16 + f] |=
                            (result[oy][ox][f * 16 + cnt] & 0b11) << (cnt * 2);
                    }
                }
            }
        }
    }

#if P25_DEBUG
    cout << Output_fm_Oy << endl;
    cout << Output_fm_Ox << endl;
    cout << cin_wr_real << endl;
    cout << para->nif << endl;
    cout << para->compare_init << endl;
    cout << para->nky << endl;
    cout << para->nkx << endl;
    cout << para->stride_y << endl;
    cout << para->stride_x << endl;

    fstream outpadx("convert_pad_x.txt", ios::out);
    for (int32_t _y = 0; _y < Py_real; ++_y) {
        for (int32_t _x = 0; _x < Px_real; ++_x) {
            for (int32_t _cin = 0; _cin < cin_wr_real; ++_cin) {
                outpadx << SI_padding[_y][_x][_cin] << endl;
            }
        }
    }
    outpadx.close();

    fstream out_array2("out_array2s.txt", ios::out);
    for (int32_t _y = 0; _y < Output_fm_Oy; ++_y) {
        for (int32_t _x = 0; _x < Output_fm_Ox; ++_x) {
            for (int32_t _cout = 0; _cout < cout_real; ++_cout) {
                out_array2 << result[_y][_x][_cout] << endl;
            }
        }
    }
    out_array2.close();

#endif

    return new_so_array;
}
