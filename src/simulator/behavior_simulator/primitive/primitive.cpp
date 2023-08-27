// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "src/simulator/behavior_simulator/primitive/primitive.h"
#include "spdlog/spdlog.h"
#include "src/simulator/behavior_simulator/data_block.h"
#include "src/simulator/behavior_simulator/identity.h"
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

void mlp(int nf, int nr, Array<int32_t, 1> &x, Array<int32_t, 2> &w,
         Array<int32_t, 1> &b, int *o) {
    for (int f = 0; f < nf; f++) {
        o[f] = b[f];
        for (int r = 0; r < nr; r++)
            o[f] = sign_cast_64_32(o[f] + (int64_t)w[r][f] * x[r]);
    }
}
void vector_add(int noy, int nox, int nf, int na, Array<int32_t, 4> &si, int *b,
                int *so, int stepy, int stepx) {
    for (int32_t oy = 0; oy < noy; oy++) {
        for (int32_t ox = 0; ox < nox; ox++) {
            for (int32_t f = 0; f < nf; f++) {
                int64_t sum = b[f];
                for (int32_t cnt = 0; cnt < na; ++cnt) {
                    sum += int64_t(si[cnt][oy][ox][f]);
                    sum = sign_cast_64_32(sum);
                }
                so[oy * stepy + ox * stepx + f] = sum;
            }
        }
    }
}
void conv2d(int noy, int nox, int nf, int nr, int nky, int nkx, int sy, int sx,
            int ey, int ex, Array<int32_t, 3> &x, Array<int32_t, 4> &w, int *b,
            int *o, int stepy, int stepx) {
    for (int32_t oy = 0; oy < noy; oy++) {
        for (int32_t ox = 0; ox < nox; ox++) {
            for (int32_t f = 0; f < nf; f++) {
                int64_t result = b[f];
                for (int32_t r = 0; r < nr; r++) {
                    for (int32_t ky = 0; ky < nky; ky++) {
                        for (int32_t kx = 0; kx < nkx; kx++) {
                            result +=
                                x[oy * sy + ey * ky][ox * sx + ex * kx][r] *
                                int64_t(w[ky][kx][r][f]);
                            result = sign_cast_64_32(result);
                        }
                    }
                }
                o[oy * stepy + ox * stepx + f] = result;
            }
        }
    }
}

void conv(int noy, int nox, int nf, int nr, int nky, int nkx, int sy, int sx,
          int ey, int ex, Array<int32_t, 3> &x, Array<int32_t, 4> &w, int *b,
          int *o, int stepy, int stepx) {
    for (int32_t oy = 0; oy < noy; oy++) {
        for (int32_t ox = 0; ox < nox; ox++) {
            for (int32_t f = 0; f < nf; f++) {
                int64_t result = b[f];
                for (int32_t r = 0; r < nr; r++) {
                    for (int32_t ky = 0; ky < nky; ky++) {
                        for (int32_t kx = 0; kx < nkx; kx++) {
                            result += int64_t(x[oy * sy + ey * ky]
                                               [ox * sx + ex * kx][r]) *
                                      int64_t(w[f][ky][kx][r]);
                            result = sign_cast_64_32(result);
                        }
                    }
                }
                o[oy * stepy + ox * stepx + f] = result;
            }
        }
    }
}
vector<int32_t> Primitive::get_output_length() const {
    auto t = get_output_shape();
    vector<int32_t> out;
    for_each(t.begin(), t.end(),
             [&](vector<int32_t> len) { out.push_back(f(len)); });
    return out;
}

size_t Primitive::f(vector<int32_t> list) const {
    int32_t len = 1;
    for (auto d : list) {
        len *= d;
    }
    return len * sizeof(uint32_t);
}
