// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "prim_06.h"
#include "src/simulator/behavior_simulator/util.h"
#include <cmath>
#include <fstream>
#include <iomanip>

#define P06_DEBUG 0

Prim06::Prim06(shared_ptr<Prim06_Parameter> para) : Primitive(SOMA, para) {
    if (para->x1_precision == 0) {
        Km_num_in = ceil(double(para->length_in) / 4.0);
        Km_num_ciso = ceil(double(para->length_ciso) / 4.0);
        length_in_equal = Km_num_in * 4;
        length_ciso_equal = Km_num_ciso * 4;
        num_in_4B = 1;
    } else if (para->x1_precision == 3) {
        Km_num_in = ceil(double(para->length_in) / 64.0);
        Km_num_ciso = ceil(double(para->length_ciso) / 64.0);
        length_in_equal = Km_num_in * 64;
        length_ciso_equal = Km_num_ciso * 64;
        num_in_4B = 16;
    } else {
        Km_num_in = ceil(double(para->length_in) / 16.0);
        Km_num_ciso = ceil(double(para->length_ciso) / 16.0);
        length_in_equal = Km_num_in * 16;
        length_ciso_equal = Km_num_ciso * 16;
        num_in_4B = 4;
    }
    if (para->out_precision == 0) {
        Km_num_out = ceil(double(para->length_out) / 4.0);
        length_out_equal = Km_num_out * 4;
    } else if (para->out_precision == 3) {
        Km_num_out = ceil(double(para->length_out) / 64.0);
        length_out_equal = Km_num_out * 64;
    } else {
        Km_num_out = ceil(double(para->length_out) / 16.0);
        length_out_equal = Km_num_out * 16;
    }
}

vector<vector<int32_t>> Prim06::get_output_shape() const {
    auto para = static_pointer_cast<Prim06_Parameter>(_para);
    if (para->out_precision == 0) {
        return {vector<int32_t>{para->num_out, length_out_equal}};
    } else if (para->out_precision == 1) {
        return {vector<int32_t>{para->num_out, length_out_equal / 4}};
    } else if (para->out_precision == 2) {
        return {vector<int32_t>{para->num_out, length_out_equal / 4}};
    } else {
        return {vector<int32_t>{para->num_out, length_out_equal / 16}};
    }
}

void Prim06::execute(const vector<DataBlock> &input,
                     vector<DataBlock> &output) const {

    auto para = static_pointer_cast<Prim06_Parameter>(_para);
    assert(output.size() == 1);

    uint32_t *p_mem_in = (uint32_t *)(input[0].get_data().get());
    uint32_t *p_mem_ciso = (uint32_t *)(input[1].get_data().get());

    auto in_array = new_array<int32_t>(para->num_in, length_in_equal);
    auto ciso_array = new_array<int32_t>(para->num_ciso, length_ciso_equal);
    int32_t **array_in = (int32_t **)in_array.get();
    int32_t **array_ciso = (int32_t **)ciso_array.get();

    int32_t pernum_in_in_4B = (Km_num_in * (16 / 4));
    int32_t pernum_ciso_in_4B = (Km_num_ciso * (16 / 4));

    int32_t real_num =
        para->real_length_in_en ? para->real_num_in : para->num_in;

    if (para->x1_precision == 0) {
        for (int32_t cnt = 0; cnt < para->num_in; ++cnt) {
            for (int32_t i = 0; i < length_in_equal / num_in_4B; ++i) {
                for (int32_t j = 0; j < num_in_4B; ++j) {
                    array_in[cnt][num_in_4B * i + j] = p_mem_in[int32_t(
                        (cnt % real_num) * pernum_in_in_4B + i)];
                }
            }
        }
        for (int32_t cnt = 0; cnt < para->num_ciso; ++cnt) {
            for (int32_t i = 0; i < length_ciso_equal / num_in_4B; ++i) {
                for (int32_t j = 0; j < num_in_4B; ++j) {
                    array_ciso[cnt][num_in_4B * i + j] =
                        p_mem_ciso[int32_t(cnt * pernum_ciso_in_4B + i)];
                }
            }
        }
    } else if (para->x1_precision == 1) {
        for (int32_t cnt = 0; cnt < para->num_in; ++cnt) {
            for (int32_t i = 0; i < length_in_equal / num_in_4B; ++i) {
                for (int32_t j = 0; j < num_in_4B; ++j) {
                    array_in[cnt][num_in_4B * i + j] =
                        int8_t(p_mem_in[int32_t(
                                   (cnt % real_num) * pernum_in_in_4B + i)] >>
                                   (8 * j) &
                               0xff);
                }
            }
        }
        for (int32_t cnt = 0; cnt < para->num_ciso; ++cnt) {
            for (int32_t i = 0; i < length_ciso_equal / num_in_4B; ++i) {
                for (int32_t j = 0; j < num_in_4B; ++j) {
                    array_ciso[cnt][num_in_4B * i + j] = int8_t(
                        p_mem_ciso[int32_t(cnt * pernum_ciso_in_4B + i)] >>
                            (8 * j) &
                        0xff);
                }
            }
        }
    } else if (para->x1_precision == 2) {
        for (int32_t cnt = 0; cnt < para->num_in; ++cnt) {
            for (int32_t i = 0; i < length_in_equal / num_in_4B; ++i) {
                for (int32_t j = 0; j < num_in_4B; ++j) {
                    array_in[cnt][num_in_4B * i + j] =
                        uint8_t(p_mem_in[int32_t(
                                    (cnt % real_num) * pernum_in_in_4B + i)] >>
                                    (8 * j) &
                                0xff);
                }
            }
        }
        for (int32_t cnt = 0; cnt < para->num_ciso; ++cnt) {
            for (int32_t i = 0; i < length_ciso_equal / num_in_4B; ++i) {
                for (int32_t j = 0; j < num_in_4B; ++j) {
                    array_ciso[cnt][num_in_4B * i + j] = uint8_t(
                        p_mem_ciso[int32_t(cnt * pernum_ciso_in_4B + i)] >>
                            (8 * j) &
                        0xff);
                }
            }
        }
    } else if (para->x1_precision == 3) {

        for (int32_t cnt = 0; cnt < para->num_in; ++cnt) {
            for (int32_t i = 0; i < length_in_equal / num_in_4B; ++i) {
                for (int32_t j = 0; j < num_in_4B; ++j) {
                    auto temp = p_mem_in[int32_t(
                                    (cnt % real_num) * pernum_in_in_4B + i)] >>
                                    (2 * j) &
                                0b11;
                    array_in[cnt][num_in_4B * i + j] = temp == 3 ? -1 : temp;
                }
            }
        }
        for (int32_t cnt = 0; cnt < para->num_ciso; ++cnt) {
            for (int32_t i = 0; i < length_ciso_equal / num_in_4B; ++i) {
                for (int32_t j = 0; j < num_in_4B; ++j) {
                    auto temp =
                        p_mem_ciso[int32_t(cnt * pernum_ciso_in_4B + i)] >>
                            (2 * j) &
                        0b11;
                    array_ciso[cnt][num_in_4B * i + j] = temp == 3 ? -1 : temp;
                }
            }
        }
    }
#if P06_DEBUG

    fstream out_x1("convert_in.txt", ios::out);
    for (int32_t cnt = 0; cnt < para->num_in; ++cnt) {
        for (int32_t i = 0; i < length_in_equal; ++i) {
            out_x1 << array_in[cnt][i] << endl;
        }
    }
    out_x1.close();

    fstream out_x2("convert_ciso.txt", ios::out);
    for (int32_t cnt = 0; cnt < para->num_ciso; ++cnt) {
        for (int32_t i = 0; i < length_ciso_equal; ++i) {
            out_x2 << array_ciso[cnt][i] << endl;
        }
    }
    out_x2.close();
#endif

    shared_ptr<int32_t> so_array = execute(array_in, array_ciso);

    output[0].set_data(reinterpret_pointer_cast<uint8_t>(so_array));
}

shared_ptr<int32_t> Prim06::execute(int32_t **array_in,
                                    int32_t **array_ciso) const {

    auto para = static_pointer_cast<Prim06_Parameter>(_para);
    shared_ptr<int32_t> so_array =
        new_array<int32_t>(para->num_out, length_out_equal);

    int32_t **array_out = (int32_t **)so_array.get();

    int32_t num_in_real = min(para->num_in, para->num_out);
    int32_t num_ciso_real = min(para->num_ciso, para->num_out);
    int32_t num_out_real = max(num_in_real, num_ciso_real);
    int32_t length_out_real =
        min((length_in_equal + length_ciso_equal), length_out_equal);

    int32_t divider = (pow(2, para->bit_shift_num * 2));
    if (para->out_precision <= para->x1_precision) {
    } else {
        for (int32_t cnt = 0; cnt < para->num_in; ++cnt) {
            for (int32_t i = 0; i < length_in_equal; ++i) {
                array_in[cnt][i] = floor(double(array_in[cnt][i]) / divider);
            }
        }
        for (int32_t cnt = 0; cnt < para->num_ciso; ++cnt) {
            for (int32_t i = 0; i < length_ciso_equal; ++i) {
                array_ciso[cnt][i] =
                    floor(double(array_ciso[cnt][i]) / divider);
            }
        }
    }
    if (para->out_precision == 1) {
        for (int32_t cnt = 0; cnt < para->num_in; ++cnt) {
            for (int32_t i = 0; i < length_in_equal; ++i) {
                if (array_in[cnt][i] > 127) {
                    array_in[cnt][i] = 127;
                } else if (array_in[cnt][i] < -128) {
                    array_in[cnt][i] = -128;
                }
            }
        }
        for (int32_t cnt = 0; cnt < para->num_ciso; ++cnt) {
            for (int32_t i = 0; i < length_ciso_equal; ++i) {
                if (array_ciso[cnt][i] > 127) {
                    array_ciso[cnt][i] = 127;
                } else if (array_ciso[cnt][i] < -128) {
                    array_ciso[cnt][i] = -128;
                }
            }
        }
    } else if (para->out_precision == 3) {
        for (int32_t cnt = 0; cnt < para->num_in; ++cnt) {
            for (int32_t i = 0; i < length_in_equal; ++i) {
                if (array_in[cnt][i] > 1) {
                    array_in[cnt][i] = 1;
                } else if (array_in[cnt][i] < -1) {
                    array_in[cnt][i] = -1;
                }
            }
        }
        for (int32_t cnt = 0; cnt < para->num_ciso; ++cnt) {
            for (int32_t i = 0; i < length_ciso_equal; ++i) {
                if (array_ciso[cnt][i] > 1) {
                    array_ciso[cnt][i] = 1;
                } else if (array_ciso[cnt][i] < -1) {
                    array_ciso[cnt][i] = -1;
                }
            }
        }
    }

    for (int32_t i = 0; i < (num_out_real); ++i) {
        for (int32_t j = 0; j < (length_out_real); ++j) {
            if (j < length_in_equal) {
                if (i < num_in_real) {
                    array_out[i][j] = array_in[i][j];
                }
            } else if (j < length_out_real) {
                if (i < num_ciso_real) {
                    array_out[i][j] = array_ciso[i][j - length_in_equal];
                }
            }
        }
    }

    shared_ptr<int32_t> new_so_array = nullptr;
    int32_t *new_array_out = nullptr;
    if (para->out_precision == 0) {
        new_so_array = new_array<int32_t>(para->num_out * length_out_equal);
        new_array_out = (int32_t *)new_so_array.get();
        for (int32_t i = 0; i < (para->num_out); ++i) {
            for (int32_t j = 0; j < (length_out_equal); ++j) {
                new_array_out[i * length_out_equal + j] = array_out[i][j];
            }
        }
    } else if (para->out_precision == 1) {
        new_so_array = new_array<int32_t>(para->num_out * length_out_equal / 4);
        new_array_out = (int32_t *)new_so_array.get();
        for (int32_t i = 0; i < (para->num_out); ++i) {
            for (int32_t j = 0; j < (length_out_equal / 4); ++j) {
                int32_t temp = 0;
                for (int32_t k = 0; k < (4); ++k) {
                    temp |= (array_out[i][j * 4 + k] & 0xff) << (k * 8);
                }
                new_array_out[i * length_out_equal / 4 + j] = temp;
            }
        }
    } else if (para->out_precision == 2) {
        new_so_array = new_array<int32_t>(para->num_out * length_out_equal / 4);
        new_array_out = (int32_t *)new_so_array.get();
        for (int32_t i = 0; i < (para->num_out); ++i) {
            for (int32_t j = 0; j < (length_out_equal / 4); ++j) {
                int32_t temp = 0;
                for (int32_t k = 0; k < (4); ++k) {
                    temp |= (array_out[i][j * 4 + k] & 0xff) << (k * 8);
                }
                new_array_out[i * length_out_equal / 4 + j] = temp;
            }
        }
    } else if (para->out_precision == 3) {
        new_so_array =
            new_array<int32_t>(para->num_out * length_out_equal / 16);
        new_array_out = (int32_t *)new_so_array.get();
        for (int32_t i = 0; i < (para->num_out); ++i) {
            for (int32_t j = 0; j < (length_out_equal / 16); ++j) {
                int32_t temp = 0;
                for (int32_t k = 0; k < (16); ++k) {
                    temp |= (array_out[i][j * 16 + k] & 0b11) << (k * 2);
                }
                new_array_out[i * length_out_equal / 16 + j] = temp;
            }
        }
    }
#if P06_DEBUG

    fstream out_x1("convert_cut_in.txt", ios::out);
    for (int32_t cnt = 0; cnt < para->num_in; ++cnt) {
        for (int32_t i = 0; i < length_in_equal; ++i) {
            out_x1 << array_in[cnt][i] << endl;
        }
    }
    out_x1.close();

    fstream out_x2("convert_cut_ciso.txt", ios::out);
    for (int32_t cnt = 0; cnt < para->num_ciso; ++cnt) {
        for (int32_t i = 0; i < length_ciso_equal; ++i) {
            out_x2 << array_ciso[cnt][i] << endl;
        }
    }
    out_x2.close();

    fstream out_o("output.txt", ios::out);
    for (int32_t i = 0; i < (para->num_out); ++i) {
        for (int32_t j = 0; j < (length_out_equal); ++j) {
            out_o << array_out[i][j] << endl;
        }
    }
    out_o.close();

    if (para->out_precision == 0) {
        fstream out_o2("output2.txt", ios::out);
        for (int32_t i = 0; i < (para->num_out * length_out_equal); ++i) {
            out_o2 << new_array_out[i] << endl;
        }
        out_o2.close();
    } else if (para->out_precision == 1) {
        fstream out_o2("output2.txt", ios::out);
        for (int32_t i = 0; i < (para->num_out * length_out_equal / 4); ++i) {
            out_o2 << new_array_out[i] << endl;
        }
        out_o2.close();
    } else if (para->out_precision == 2) {
        fstream out_o2("output2.txt", ios::out);
        for (int32_t i = 0; i < (para->num_out * length_out_equal / 4); ++i) {
            out_o2 << new_array_out[i] << endl;
        }
        out_o2.close();
    } else if (para->out_precision == 3) {
        fstream out_o2("output2.txt", ios::out);
        for (int32_t i = 0; i < (para->num_out * length_out_equal / 16); ++i) {
            out_o2 << new_array_out[i] << endl;
        }
        out_o2.close();
    }

#endif
    return new_so_array;
}
