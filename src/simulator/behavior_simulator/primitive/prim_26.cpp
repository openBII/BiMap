// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "prim_26.h"
#include "src/simulator/behavior_simulator/util.h"
#include <cmath>
#include <fstream>
#include <iomanip>

#define P26_DEBUG 0

Prim26::Prim26(shared_ptr<Prim26_Parameter> para) : Primitive(SOMA, para) {
    if (para->x1_precision == 0) {
        Km_num_in = ceil(double(para->length_in) / 4.0);
        length_in_equal = Km_num_in * 4;
        num_in_4B = 1;
    } else if (para->x1_precision == 3) {
        Km_num_in = ceil(double(para->length_in) / 64.0);
        length_in_equal = Km_num_in * 64;
        num_in_4B = 16;
    } else {
        Km_num_in = ceil(double(para->length_in) / 16.0);
        length_in_equal = Km_num_in * 16;
        num_in_4B = 4;
    }
    if (para->out_precision == 0) {
        Km_num_out = ceil(double(para->length_out) / 4.0);
        length_out_equal = Km_num_out * 4;

        Km_num_ciso = ceil(double(para->length_ciso) / 4.0);
        length_ciso_equal = Km_num_ciso * 4;
    } else if (para->out_precision == 3) {
        Km_num_out = ceil(double(para->length_out) / 64.0);
        length_out_equal = Km_num_out * 64;

        Km_num_ciso = ceil(double(para->length_ciso) / 64.0);
        length_ciso_equal = Km_num_ciso * 64;
    } else {
        Km_num_out = ceil(double(para->length_out) / 16.0);
        length_out_equal = Km_num_out * 16;

        Km_num_ciso = ceil(double(para->length_ciso) / 16.0);
        length_ciso_equal = Km_num_ciso * 16;
    }

    num_out_real = max(para->num_out, para->num_ciso);
}

vector<vector<int32_t>> Prim26::get_output_shape() const {
    auto para = static_pointer_cast<Prim26_Parameter>(_para);
    if (para->out_precision == 0) {
        return {{para->num_out, length_out_equal},
                {para->num_ciso, length_ciso_equal}};
    } else if (para->out_precision == 1) {
        return {{para->num_out, length_out_equal / 4},
                {para->num_ciso, length_ciso_equal / 4}};
    } else if (para->out_precision == 2) {
        return {{para->num_out, length_out_equal / 4},
                {para->num_ciso, length_ciso_equal / 4}};
    } else {
        return {{para->num_out, length_out_equal / 16},
                {para->num_ciso, length_ciso_equal / 16}};
    }
}

void Prim26::execute(const vector<DataBlock> &input,
                     vector<DataBlock> &output) const {

    auto para = static_pointer_cast<Prim26_Parameter>(_para);
    assert(output.size() == 2);

    uint32_t *p_mem_in = (uint32_t *)(input[0].get_data().get());

#if P26_DEBUG

    fstream t_x1("c_raw_in_mem.txt", ios::out);
    for (int32_t cnt = 0; cnt < 1215; ++cnt) {
        t_x1 << int32_t(p_mem_in[cnt]) << endl;
    }
    t_x1.close();

#endif
    auto in_array = new_array<int32_t>(para->num_in, length_in_equal);
    int32_t **array_in = (int32_t **)in_array.get();

    int32_t pernum_in_in_4B = (Km_num_in * (16 / 4));

    if (para->x1_precision == 0) {
        for (int32_t cnt = 0; cnt < para->num_in; ++cnt) {
            for (int32_t i = 0; i < length_in_equal / num_in_4B; ++i) {
                for (int32_t j = 0; j < num_in_4B; ++j) {
                    array_in[cnt][num_in_4B * i + j] =
                        p_mem_in[int32_t(cnt * pernum_in_in_4B + i)];
                }
            }
        }
    } else if (para->x1_precision == 1) {
        for (int32_t cnt = 0; cnt < para->num_in; ++cnt) {
            for (int32_t i = 0; i < length_in_equal / num_in_4B; ++i) {
                for (int32_t j = 0; j < num_in_4B; ++j) {
                    array_in[cnt][num_in_4B * i + j] =
                        int8_t(p_mem_in[int32_t(cnt * pernum_in_in_4B + i)] >>
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
                        uint8_t(p_mem_in[int32_t(cnt * pernum_in_in_4B + i)] >>
                                    (8 * j) &
                                0xff);
                }
            }
        }
    } else if (para->x1_precision == 3) {
        for (int32_t cnt = 0; cnt < para->num_in; ++cnt) {
            for (int32_t i = 0; i < length_in_equal / num_in_4B; ++i) {
                for (int32_t j = 0; j < num_in_4B; ++j) {
                    auto temp = p_mem_in[int32_t(cnt * pernum_in_in_4B + i)] >>
                                    (2 * j) &
                                0b11;
                    array_in[cnt][num_in_4B * i + j] = temp == 3 ? -1 : temp;
                }
            }
        }
    }
#if P26_DEBUG

    fstream out_x1("c_raw_in.txt", ios::out);
    for (int32_t cnt = 0; cnt < para->num_in; ++cnt) {
        for (int32_t i = 0; i < length_in_equal; ++i) {
            out_x1 << array_in[cnt][i] << endl;
        }
    }
    out_x1.close();

#endif
    auto output_ptr = execute(array_in);
    output[0].set_data(reinterpret_pointer_cast<uint8_t>(output_ptr[0]));
    output[1].set_data(reinterpret_pointer_cast<uint8_t>(output_ptr[1]));
}

vector<shared_ptr<int32_t>> Prim26::execute(int32_t **array_in) const {

    auto para = static_pointer_cast<Prim26_Parameter>(_para);

    shared_ptr<int32_t> so_array =
        new_array<int32_t>(num_out_real, length_out_equal);
    int32_t **array_out = (int32_t **)so_array.get();

    shared_ptr<int32_t> ciso_array =
        new_array<int32_t>(num_out_real, length_ciso_equal);
    int32_t **array_ciso = (int32_t **)ciso_array.get();

    int32_t num_in_out_real = min(para->num_in, para->num_out);
    int32_t num_in_ciso_real = min(para->num_in, para->num_ciso);
    int32_t length_out_real =
        min((length_in_equal), (length_out_equal + length_ciso_equal));

    int32_t divider = (pow(2, para->bit_shift_num * 2));

    if (para->out_precision > para->x1_precision) {
        for (int32_t cnt = 0; cnt < para->num_in; ++cnt) {
            for (int32_t i = 0; i < length_in_equal; ++i) {
                array_in[cnt][i] = floor(double(array_in[cnt][i]) / divider);
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
    }

    for (int32_t i = 0; i < (num_out_real); ++i) {
        for (int32_t j = 0; j < (length_out_real); ++j) {
            if (j < length_out_equal) {
                if (i < num_in_out_real) {
                    array_out[i][j] = array_in[i][j];
                }
            } else if (j < length_out_real) {
                if (i < num_in_ciso_real) {
                    array_ciso[i][j - length_out_equal] = array_in[i][j];
                }
            }
        }
    }

    shared_ptr<int32_t> new_so_array = nullptr;
    int32_t *new_array_out = nullptr;
    shared_ptr<int32_t> new_ciso_array = nullptr;
    int32_t *new_array_ciso = nullptr;
    if (para->out_precision == 0) {
        new_so_array = new_array<int32_t>(para->num_out * length_out_equal);
        new_array_out = (int32_t *)new_so_array.get();
        for (int32_t i = 0; i < (para->num_out); ++i) {
            for (int32_t j = 0; j < (length_out_equal); ++j) {
                new_array_out[i * length_out_equal + j] = array_out[i][j];
            }
        }
        new_ciso_array = new_array<int32_t>(para->num_ciso * length_ciso_equal);
        new_array_ciso = (int32_t *)new_ciso_array.get();
        for (int32_t i = 0; i < (para->num_ciso); ++i) {
            for (int32_t j = 0; j < (length_ciso_equal); ++j) {
                new_array_ciso[i * length_ciso_equal + j] = array_ciso[i][j];
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
        new_ciso_array =
            new_array<int32_t>(para->num_ciso * length_ciso_equal / 4);
        new_array_ciso = (int32_t *)new_ciso_array.get();
        for (int32_t i = 0; i < (para->num_ciso); ++i) {
            for (int32_t j = 0; j < (length_ciso_equal / 4); ++j) {
                int32_t temp = 0;
                for (int32_t k = 0; k < (4); ++k) {
                    temp |= (array_ciso[i][j * 4 + k] & 0xff) << (k * 8);
                }
                new_array_ciso[i * length_ciso_equal / 4 + j] = temp;
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
        new_ciso_array =
            new_array<int32_t>(para->num_ciso * length_ciso_equal / 4);
        new_array_ciso = (int32_t *)new_ciso_array.get();
        for (int32_t i = 0; i < (para->num_ciso); ++i) {
            for (int32_t j = 0; j < (length_ciso_equal / 4); ++j) {
                int32_t temp = 0;
                for (int32_t k = 0; k < (4); ++k) {
                    temp |= (array_ciso[i][j * 4 + k] & 0xff) << (k * 8);
                }
                new_array_ciso[i * length_ciso_equal / 4 + j] = temp;
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
        new_ciso_array =
            new_array<int32_t>(para->num_ciso * length_ciso_equal / 16);
        new_array_ciso = (int32_t *)new_ciso_array.get();
        for (int32_t i = 0; i < (para->num_ciso); ++i) {
            for (int32_t j = 0; j < (length_ciso_equal / 16); ++j) {
                int32_t temp = 0;
                for (int32_t k = 0; k < (16); ++k) {
                    temp |= (array_ciso[i][j * 16 + k] & 0b11) << (k * 2);
                }
                new_array_ciso[i * length_ciso_equal / 16 + j] = temp;
            }
        }
    }
#if P26_DEBUG

    fstream out_array("out_array.txt", ios::out);
    for (int32_t i = 0; i < (num_out_real); ++i) {
        for (int32_t j = 0; j < (length_out_equal); ++j) {
            out_array << array_out[i][j] << endl;
        }
    }
    out_array.close();
    fstream f_ciso_array("ciso_array.txt", ios::out);
    for (int32_t i = 0; i < (num_out_real); ++i) {
        for (int32_t j = 0; j < (length_ciso_equal); ++j) {
            f_ciso_array << array_ciso[i][j] << endl;
        }
    }
    f_ciso_array.close();

    fstream out_x1("convert_cut_in.txt", ios::out);
    for (int32_t cnt = 0; cnt < para->num_in; ++cnt) {
        for (int32_t i = 0; i < length_in_equal; ++i) {
            out_x1 << array_in[cnt][i] << endl;
        }
    }
    out_x1.close();

#endif
    return {new_so_array, new_ciso_array};
}
