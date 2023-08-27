// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "src/simulator/behavior_simulator/data_block.h"
#include "src/simulator/behavior_simulator/identity.h"
#include "src/simulator/behavior_simulator/patch.h"
#include "src/simulator/behavior_simulator/util.h"
#include <memory>
#include <vector>

using namespace std;

void vector_add(int noy, int nox, int nf, int na, Array<int32_t, 4> &si, int *b,
                int *so, int stepy, int stepx);
void conv(int noy, int nox, int nf, int nr, int nky, int nkx, int sy, int sx,
          int ey, int ex, Array<int32_t, 3> &x, Array<int32_t, 4> &w, int *b,
          int *o, int stepy, int stepx);
void conv2d(int noy, int nox, int nf, int nr, int nky, int nkx, int sy, int sx,
            int ey, int ex, Array<int32_t, 3> &x, Array<int32_t, 4> &w, int *b,
            int *o, int stepy, int stepx);
void mlp(int nf, int nr, Array<int32_t, 1> &x, Array<int32_t, 2> &w,
         Array<int32_t, 1> &b, int *o);
struct Parameter {
};

class Primitive
{
 public:
    enum TYPE { AXON, SOMA, ROUTER };
    Primitive(TYPE t, shared_ptr<Parameter> para = nullptr)
        : type(t), _para(para) {}
    virtual void execute(const vector<DataBlock> &input,
                         vector<DataBlock> &output) const {}
    TYPE get_type() const { return type; }
    ~Primitive() {}
    shared_ptr<Parameter> get_parameters() const { return _para; }
    vector<ID> get_input_id_list() const { return input_list; }
    void add_input_id(const ID &id) { input_list.push_back(id); }
    void add_output_id(const ID &id) { output_list.push_back(id); }
    vector<ID> get_output_id_list() const { return output_list; }
    virtual vector<vector<int32_t>> get_output_shape() const {
        return vector<vector<int32_t>>();
    }
    vector<int32_t> get_output_length() const;

 protected:
    size_t f(vector<int32_t> list) const;
    TYPE type;
    shared_ptr<Parameter> _para;
    vector<ID> input_list;
    vector<ID> output_list;
};

#endif  // PRIMITIVE_H
