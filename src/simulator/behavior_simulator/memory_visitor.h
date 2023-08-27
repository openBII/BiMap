// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#ifndef MEMORYVISITOR_H
#define MEMORYVISITOR_H

#include "top/global_config.h"

#include "src/simulator/behavior_simulator/context.h"
#include "src/simulator/behavior_simulator/identity.h"

#include <iomanip>
#include <map>

using std::pair;
using std::string;

class MemoryVisitor
{
 public:
    void segment_output(size_t start, size_t length, ostream &s, uint32_t *p,
                        const size_t &cur) {

        if (length == 0)
            return;
        size_t remain = 0x4000 - start % 0x4000;
        remain = remain > length ? length : remain;
        s << hex;

        if (GlobalConfig::TEST_MODE == tianjic_ir::PRIM_OUTPUT) {
            s << setw(8) << setfill('0') << start << endl;
            s << setw(8) << setfill('0') << remain << endl;
        }

        for (size_t n = 0; n < remain; n++) {
            s << setw(8) << setfill('0') << p[cur + n] << endl;
        }

        segment_output(start + remain, length - remain, s, p, cur + remain);
    }

    void add_output_segment(pair<size_t, size_t> seg) {
        if (seg.first != MEM_SIZE) {
            seg_pool.push_back(seg);
        }
    }
    void add_output_segment(pair<size_t, size_t> seg, std::string id) {
        if (seg.first != MEM_SIZE) {
            seg_pool.push_back(seg);
            id_pool[seg] = id;
        }
    }

    void serialize_fstream(const ID &core_id, int32_t cx, int32_t cy,
                           int32_t group_id, int32_t x, int32_t y,
                           int32_t step_num, size_t phase_num,
                           shared_ptr<Context> ctx) {

        char filename[128];
        sort(seg_pool.begin(), seg_pool.end(),
             [](const pair<size_t, size_t> &seg1,
                const pair<size_t, size_t> &seg2) {
                 return seg1.first < seg2.first;
             });

        if (GlobalConfig::TEST_MODE == tianjic_ir::PRIM_OUTPUT) {
            sprintf(filename, "%s/cmp_out_%d_%d_%d_%d_%d@%d_%ld.txt",
                    GlobalConfig::OUTPUT_DIR.c_str(), cx, cy, group_id, x, y,
                    step_num, phase_num);
            fstream f(filename, ios::out);
            for_each(seg_pool.begin(), seg_pool.end(),
                     [&](const pair<size_t, size_t> &seg) {
                         size_t start = seg.first;
                         size_t length = seg.second;
                         f << hex;
                         if (length == 0) {
                             f << setw(8) << setfill('0') << start / 4 << endl;
                             f << setw(8) << setfill('0') << 0 << endl;
                         } else {
                             auto ptr = ctx->read(core_id, start, length);
                             segment_output(start / 4, length / 4, f,
                                            (uint32_t *)ptr.get(), 0);
                         }
                     });
            f.close();
        } else {
            for_each(seg_pool.begin(), seg_pool.end(),
                     [&](const pair<size_t, size_t> &seg) {
                         sprintf(filename, "%s/%s.hex",
                                 GlobalConfig::OUTPUT_DIR.c_str(),
                                 id_pool[seg].c_str());
                         fstream f(filename, ios::out);
                         size_t start = seg.first;
                         size_t length = seg.second;
                         f << hex;
                         if (length) {
                             auto ptr = ctx->read(core_id, start, length);
                             segment_output(start / 4, length / 4, f,
                                            (uint32_t *)ptr.get(), 0);
                         }
                         f.close();
                     });
        }
    }

 private:
    vector<pair<size_t, size_t>> seg_pool;
    map<pair<size_t, size_t>, std::string> id_pool;
};

class MemoryVisitorMaster
{
 public:
    void set_visitor(const ID &core_id, size_t phase_num,
                     MemoryVisitor visitor) {
        visitor_pool[core_id][phase_num] = visitor;
    }

    map<size_t, MemoryVisitor> get_visitor_map(const ID &core_id) {
        return visitor_pool[core_id];
    }

 private:
    map<ID, map<size_t, MemoryVisitor>> visitor_pool;
};

#endif  // MEMORYVISITOR_H
