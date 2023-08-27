// Copyright (C) OpenBII
// Team: CBICR
// SPDX-License-Identifier: Apache-2.0
// See: https://spdx.org/licenses/

#include "prim_08.h"
#include "src/simulator/behavior_simulator/util.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

#define P08_INPUT_DEBUG 0
// VM
// V_theta
// ref_cnt
// Uin
// V
#define P08_OUTPUT_DEBUG 0

// lfsr_vec
// lfsr_mask
// lfsr_mask_cut
// vth
// V
// spike
// V_update

// V_new
// V_theta
// ref_cnt
// new_out_reg

lfsr::lfsr(int32_t seed) : _lfsr(seed) {
    if (_lfsr < 0) {
        _lfsr += int64_t(4294967296);
    }
}
int64_t lfsr::update() {
    int64_t bit = ((_lfsr >> 0) ^ (_lfsr >> 1) ^ (_lfsr >> 2) ^ (_lfsr >> 3) ^
                   (_lfsr >> 5) ^ (_lfsr >> 7)) &
                  1;
    _lfsr = (bit << 31) | (_lfsr >> 1);

    if (_lfsr > (int64_t(2147483647))) {
        return _lfsr - int64_t(4294967296);
    } else {
        return _lfsr;
    }
}

Prim08::Prim08(shared_ptr<Prim08_Parameter> para)
    : Primitive(SOMA, para), _lfsr(lfsr(para->seed)) {

    neuron_length = para->neuron_num * para->group_num;
}

vector<vector<int32_t>> Prim08::get_output_shape() const {
    auto para = static_pointer_cast<Prim08_Parameter>(_para);
    int32_t shape_1 = para->neuron_num;
    if ((para->fire_type == 2) || (para->fire_type == 3) ||
        (para->fire_type == 4) || (para->fire_type == 7)) {
        shape_1 = ceil(double(shape_1) / 16.0) * 16;
    } else if (para->fire_type == 5) {
        shape_1 = ceil(double(shape_1) / 64.0) * 64;
    }

    if ((para->fire_type == 1) || (para->fire_type == 0) ||
        (para->fire_type == 6)) {
        return {{para->group_num, shape_1},
                {neuron_length},
                {para->Vtheta_len},
                {4}};
    } else if ((para->fire_type == 2) || (para->fire_type == 3) ||
               (para->fire_type == 4) || (para->fire_type == 7)) {
        return {{para->group_num, shape_1 / 4},
                {neuron_length},
                {para->Vtheta_len},
                {4}};
    } else if (para->fire_type == 5) {
        return {{para->group_num, shape_1 / 16},
                {neuron_length},
                {para->Vtheta_len},
                {4}};
    } else {
        throw runtime_error("fire_type error\n");
    }
}

void Prim08::execute(const vector<DataBlock> &input,
                     vector<DataBlock> &output) const {

    auto para = static_pointer_cast<Prim08_Parameter>(_para);
    assert(output.size() == 4);

    uint32_t *p_mem_Uin = (uint32_t *)(input[0].get_data().get());
    uint32_t *p_mem_VM_para = (uint32_t *)(input[1].get_data().get());
    uint32_t *p_mem_v_ref_cnt = (uint32_t *)(input[2].get_data().get());
    uint32_t *p_mem_Vtheta_para = (uint32_t *)(input[3].get_data().get());

    int32_t VM_repeat = neuron_length / para->VM_len;
    int32_t Vtheta_repeat = neuron_length / para->Vtheta_len;

    auto p_Uin = new_array<int32_t>(neuron_length);
    int32_t *Uin = p_Uin.get();

    auto p_VM = new_array<int32_t>(para->VM_len * VM_repeat);
    int32_t *VM = p_VM.get();

    auto p_Vtheta = new_array<int32_t>(para->Vtheta_len * Vtheta_repeat);
    int32_t *Vtheta = p_Vtheta.get();

    auto p_V = new_array<int32_t>(neuron_length);
    int32_t *V = p_V.get();

    auto p_ref_cnt = new_array<int32_t>(neuron_length);
    int32_t *ref_cnt = p_ref_cnt.get();

    for (int32_t i = 0; i < neuron_length; ++i) {
        Uin[i] = p_mem_Uin[i];
    }

    for (int32_t j = 0; j < VM_repeat; ++j) {
        for (int32_t i = 0; i < para->VM_len; ++i) {
            VM[j * para->VM_len + i] = p_mem_VM_para[i];
        }
    }

    for (int32_t j = 0; j < Vtheta_repeat; ++j) {
        for (int32_t i = 0; i < para->Vtheta_len; ++i) {
            Vtheta[j * para->Vtheta_len + i] = p_mem_Vtheta_para[i];
        }
    }

    for (int32_t i = 0; i < neuron_length; ++i) {
        V[i] = p_mem_v_ref_cnt[i] & 268435455;
        if (V[i] > 0x7ffffff) {
            V[i] = (V[i] | 0xf0000000) - 0xffffffff - 1;
        }
        ref_cnt[i] = (p_mem_v_ref_cnt[i] & 0xffffffff) >> 28;
    }

#if P08_INPUT_DEBUG
    fstream in_VM("c_in_VM.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        in_VM << VM[i] << endl;
    }
    in_VM.close();

    fstream in_Vtheta("c_in_Vtheta.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        in_Vtheta << Vtheta[i] << endl;
    }
    in_Vtheta.close();

    fstream in_ref_cnt("c_in_ref_cnt.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        in_ref_cnt << ref_cnt[i] << endl;
    }
    in_ref_cnt.close();

    fstream in_Uin("c_in_Uin.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        in_Uin << Uin[i] << endl;
    }
    in_Uin.close();

    fstream in_V("c_in_V.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        in_V << V[i] << endl;
    }
    in_V.close();
#endif
    auto result = execute(VM, Vtheta, V, ref_cnt, Uin);

    for (int32_t i = 0; i < 4; ++i) {
        output[i].set_data(reinterpret_pointer_cast<uint8_t>(result[i]));
    }
}

vector<shared_ptr<int32_t>> Prim08::execute(int32_t *VM, int32_t *Vtheta,
                                            int32_t *V, int32_t *ref_cnt,
                                            int32_t *Uin) const {

    auto para = static_pointer_cast<Prim08_Parameter>(_para);
    bool Tw_finish = para->Tw_cnt >= para->Tw_len;

    vector<int32_t> lfsr_vec(neuron_length);
    for (int32_t i = 0; i < neuron_length; ++i) {
        lfsr_vec[i] = _lfsr.update();
    }
#if P08_OUTPUT_DEBUG
    fstream out_lfsr_vec("c_out_lfsr_vec.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        out_lfsr_vec << lfsr_vec[i] << endl;
    }
    out_lfsr_vec.close();
#endif

    vector<int32_t> lfsr_mask(neuron_length);
    for (int32_t i = 0; i < neuron_length; ++i) {
        if (lfsr_vec[i] < 0) {
            lfsr_mask[i] = ~((~lfsr_vec[i]) & VM[i]) + 1;
        } else {
            lfsr_mask[i] = lfsr_vec[i] & VM[i];
        }
    }
#if P08_OUTPUT_DEBUG
    fstream out_lfsr_mask("c_out_lfsr_mask.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        out_lfsr_mask << lfsr_mask[i] << endl;
    }
    out_lfsr_mask.close();
#endif

    vector<int32_t> lfsr_mask_cut(neuron_length);
    for (int32_t i = 0; i < neuron_length; ++i) {
        lfsr_mask_cut[i] = lfsr_mask[i] & 0xfffffff;
    }
    for (int32_t i = 0; i < neuron_length; ++i) {
        if (lfsr_mask_cut[i] > 0x7ffffff) {
            lfsr_mask_cut[i] -= 0x10000000;
        }
    }

    for (int32_t i = 0; i < neuron_length; ++i) {
        if (lfsr_mask_cut[i] > 0x7ffffff) {
            lfsr_mask_cut[i] -= 0x10000000;
        }
    }

#if P08_OUTPUT_DEBUG
    fstream out_lfsr_mask_cut("c_out_lfsr_mask_cut.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        out_lfsr_mask_cut << lfsr_mask_cut[i] << endl;
    }
    out_lfsr_mask_cut.close();
#endif

    vector<int32_t> Vth(neuron_length);
    for (int32_t i = 0; i < neuron_length; ++i) {
        Vth[i] = para->Vth0 + Vtheta[i] + lfsr_mask_cut[i];

        //            self.Vth.clip(-2 ** 27, 2 ** 27 - 1)  # 28bit饱和截取
        if (Vth[i] > 134217727) {
            Vth[i] = 134217727;
        } else if (Vth[i] < -134217728) {
            Vth[i] = -134217728;
        }

        //            self.Vth[self.Vth < self.VL] = self.VL  # 下限饱和
        if (Vth[i] < para->VL) {
            Vth[i] = para->VL;
        }
    }
#if P08_OUTPUT_DEBUG
    fstream out_vth("c_out_vth.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        out_vth << Vth[i] << endl;
    }
    out_vth.close();

    fstream in_Uin("c_pre_cal_V_Uin.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        in_Uin << Uin[i] << endl;
    }
    in_Uin.close();

    fstream in_V("c_pre_cal_V_V.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        in_V << V[i] << endl;
    }
    in_V.close();

    fstream in_ref_cnt("c_pre_cal_V_ref_cnt.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        in_ref_cnt << ref_cnt[i] << endl;
    }
    in_ref_cnt.close();
#endif

    vector<int64_t> intermediate_v(neuron_length);
    for (int32_t i = 0; i < neuron_length; ++i) {
        //            self.V[self.ref_cnt == 0] = self.V[self.ref_cnt == 0] +
        //            self.Uin[self.ref_cnt == 0]
        if (ref_cnt[i] == 0) {
            intermediate_v[i] = int64_t(Uin[i]) + int64_t(V[i]);
        } else {
            intermediate_v[i] = int64_t(V[i]);
        }
        //            self.V = self.V.clip(-2 ** 27, 2 ** 27 - 1)
        //            if(V[i]>134217727){
        //                V[i]=134217727;
        //            }
        //            else if(V[i]<-134217728){
        //                V[i]=-134217728;
        //            }
    }

#if P08_OUTPUT_DEBUG
    fstream out_iV("c_out_intermediate_V.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        out_iV << intermediate_v[i] << endl;
    }
    out_iV.close();
#endif
    for (int32_t i = 0; i < neuron_length; ++i) {
        if (intermediate_v[i] > 134217727) {
            intermediate_v[i] = 134217727;
        } else if (intermediate_v[i] < -134217728) {
            intermediate_v[i] = -134217728;
        }
    }
    //        cout<<(1<<27)-1;
    //        cout<<-(1<<27);
    for (int32_t i = 0; i < neuron_length; ++i) {
        V[i] = intermediate_v[i];
    }
#if P08_OUTPUT_DEBUG
    fstream out_V("c_out_V.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        out_V << V[i] << endl;
    }
    out_V.close();
#endif

    vector<bool> spike(neuron_length, false);
    //        spike = self.V > self.Vth
    for (int32_t i = 0; i < neuron_length; ++i) {
        if (V[i] > Vth[i]) {
            spike[i] = true;
        }
    }
#if P08_OUTPUT_DEBUG
    fstream out_spike("c_out_spike.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        out_spike << spike[i] << endl;
    }
    out_spike.close();
#endif

    //        V_update = np.zeros(self.neuron_length).astype('int32')
    //        V_update[:] = self.V[:]
    vector<int32_t> V_update(V, V + neuron_length);
    /*
         * if self.reset_mode == 0:
            V_update[spike == 1] = self.VR
        elif self.reset_mode == 1:
            V_update[spike == 1] = V_update[spike == 1] - self.Vth[spike == 1]
        elif self.reset_mode == 2:
            V_update[spike == 1] = V_update[spike == 1] - self.dV
        else:
            V_update = V_update
         */

    for (int32_t i = 0; i < neuron_length; ++i) {
        if (spike[i]) {
            if (para->reset_mode == 0) {
                V_update[i] = para->VR;
            } else if (para->reset_mode == 1) {
                V_update[i] = V_update[i] - Vth[i];
            } else if (para->reset_mode == 2) {
                V_update[i] = V_update[i] - para->dV;
            }
        }
    }
#if P08_OUTPUT_DEBUG
    fstream out_V_update("c_out_V_update.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        out_V_update << V_update[i] << endl;
    }
    out_V_update.close();
#endif

    //        V_new = np.zeros(self.neuron_length).astype('int32')
    vector<int32_t> V_new(neuron_length, para->Vinit);
    /*
         * if Tw_finish is False:
            if self.Vleaky_adpt_en is True:  # 自适应开启
                V_new[:] = (V_update.astype('int64') *
                            self.Vleaky_alpha // 256) + self.Vleaky_beta
            else:  # 常值变化
                V_new[:] = V_update + self.Vleaky_beta
        else:  # 时间窗结束，复位成初始值
            V_new[:] = self.Vinit
         */
    if (Tw_finish == false) {
        //            cout<<"Tw_finish==false"<<endl;
        if (para->Vleaky_adpt_en) {
            for (int32_t i = 0; i < neuron_length; ++i) {
                V_new[i] =
                    floor(double(int64_t(V_update[i]) * para->Vleaky_alpha) /
                          256.0) +
                    para->Vleaky_beta;
            }
        } else {
            for (int32_t i = 0; i < neuron_length; ++i) {
                V_new[i] = V_update[i] + para->Vleaky_beta;
            }
        }
    }

    //        V_new = V_new.clip(-2 ** 27, 2 ** 27 - 1)
    //        V_new[V_new < self.VL] = self.VL
    for (int32_t i = 0; i < neuron_length; ++i) {
        if (V_new[i] > 134217727) {
            V_new[i] = 134217727;
        } else if (V_new[i] < -134217728) {
            V_new[i] = -134217728;
        }
        if (V_new[i] < para->VL) {
            V_new[i] = para->VL;
        }
    }
#if P08_OUTPUT_DEBUG
    fstream out_V_new("c_out_V_new.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        out_V_new << V_new[i] << endl;
    }
    out_V_new.close();
#endif

    /*
         * if self.Vth_adpt_en is True:
            Vtheta_updata = (self.Vtheta.astype('int64') *
                             self.Vth_alpha // 256) + self.Vth_beta
        else:
            Vtheta_updata = self.Vtheta + self.Vth_beta
         */

    vector<int32_t> Vtheta_updata(neuron_length);
    if (para->Vth_adpt_en) {
        for (int32_t i = 0; i < neuron_length; ++i) {
            //                double
            //                step1=double(int64_t(Vtheta[i])*para->Vth_alpha);
            //                int32_t step2=floor(step1/256.0);
            //                cout<<step1<<" "<<step2<<endl;
            //                system("pause");
            //                Vtheta_updata[i]=step2+para->Vth_beta;
            Vtheta_updata[i] =
                floor(double(int64_t(Vtheta[i]) * para->Vth_alpha) / 256.0) +
                para->Vth_beta;
        }
    } else {
        for (int32_t i = 0; i < neuron_length; ++i) {
            Vtheta_updata[i] = Vtheta[i] + para->Vth_beta;
        }
    }

    //        Vtheta_updata[spike == 1] = Vtheta_updata[spike == 1] +
    //        self.Vth_Incre
    for (int32_t i = 0; i < neuron_length; ++i) {
        if (spike[i] == 1) {
            Vtheta_updata[i] += para->Vth_Incre;
        }
    }
    //        Vtheta_updata = Vtheta_updata.clip(-2 ** 27, 2 ** 27 - 1)
    for (int32_t i = 0; i < neuron_length; ++i) {
        if (Vtheta_updata[i] > 134217727) {
            Vtheta_updata[i] = 134217727;
        } else if (Vtheta_updata[i] < -134217728) {
            Vtheta_updata[i] = -134217728;
        }
    }

#if P08_OUTPUT_DEBUG
    fstream out_Vtheta_updata("c_out_Vtheta_updata.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        out_Vtheta_updata << Vtheta_updata[i] << endl;
    }
    out_Vtheta_updata.close();
#endif

    /*
         * if self.fire_type == 0:
            self.out_reg[:] = self.V[:]
        elif self.fire_type == 1:
            self.out_reg[:] = V_new[:]
        elif self.fire_type == 2:
            self.out_reg[:] = self.V[:] // 2 ** (self.bit_shift_num * 2)
            self.out_reg = self.out_reg.clip(-128, 127)
        elif self.fire_type == 3:
            self.out_reg[:] = V_new[:] // 2 ** (self.bit_shift_num * 2)
            self.out_reg = self.out_reg.clip(-128, 127)
        elif self.fire_type == 4 or self.fire_type == 5:
            self.out_reg[:] = spike
        elif self.fire_type == 6:
            self.out_reg[:] = lfsr_mask
        elif self.fire_type == 7:
            self.out_reg[:] = np.bitwise_and(lfsr_mask[:], 255)
            self.out_reg[self.out_reg >
                         127] = self.out_reg[self.out_reg > 127] - 256
         */

    int32_t divider = (2 << (para->bit_shift_num * 2));
    vector<int32_t> out_reg(neuron_length);
    if (para->fire_type == 0) {
        out_reg = vector<int32_t>(V, V + neuron_length);
    } else if (para->fire_type == 1) {
        out_reg = V_new;
    } else if (para->fire_type == 2) {
        out_reg = vector<int32_t>(V, V + neuron_length);

        for (int32_t i = 0; i < neuron_length; ++i) {
            out_reg[i] /= divider;
            if (out_reg[i] > 127) {
                out_reg[i] = 127;
            } else if (out_reg[i] < (-128)) {
                out_reg[i] = -128;
            }
        }
    } else if (para->fire_type == 3) {
        for (int32_t i = 0; i < neuron_length; ++i) {
            out_reg[i] = V_new[i] / divider;
            if (out_reg[i] > 127) {
                out_reg[i] = 127;
            } else if (out_reg[i] < (-128)) {
                out_reg[i] = -128;
            }
        }
    } else if (para->fire_type == 4) {
        for (int32_t i = 0; i < neuron_length; ++i) {
            out_reg[i] = spike[i];
        }
    } else if (para->fire_type == 5) {
        for (int32_t i = 0; i < neuron_length; ++i) {
            out_reg[i] = spike[i];
        }
    } else if (para->fire_type == 6) {
        for (int32_t i = 0; i < neuron_length; ++i) {
            out_reg[i] = lfsr_mask[i];
        }
    } else if (para->fire_type == 7) {
        for (int32_t i = 0; i < neuron_length; ++i) {
            out_reg[i] = lfsr_mask[i] & 0xff;
            if (out_reg[i] > 127) {
                out_reg[i] -= 256;
            }
        }
    }

    /*
         * for i in range(self.neuron_length):
            if spike[i] == 1:
                self.ref_cnt[i] = self.Ref_len
            elif self.ref_cnt[i] > 0:
                self.ref_cnt[i] = self.ref_cnt[i] - 1
            else:
                self.ref_cnt[i] = 0
         */
    for (int32_t i = 0; i < neuron_length; ++i) {
        if (spike[i] == 1) {
            ref_cnt[i] = para->Ref_len;
        } else if (ref_cnt[i] > 0) {
            --ref_cnt[i];
        } else {
            ref_cnt[i] = 0;
        }
    }

#if P08_OUTPUT_DEBUG
    fstream out_ref_cnt("c_out_ref_cnt.txt", ios::out);
    for (int32_t i = 0; i < neuron_length; ++i) {
        out_ref_cnt << ref_cnt[i] << endl;
    }
    out_ref_cnt.close();
#endif

    /*
         * if self.Tw_en is True:
            if self.Tw_cnt == self.Tw_len:
                self.Tw_cnt = 0
            else:
                self.Tw_cnt = self.Tw_cnt + 1
         */

    int32_t &Tw_cnt_new = para->Tw_cnt;

    if (para->Tw_en) {
        if (para->Tw_cnt == para->Tw_len) {
            Tw_cnt_new = 0;
        } else {
            ++Tw_cnt_new;
        }
    }

    /*
         * self.out_reg = self.out_reg.reshape(self.group_num, self.neuron_num)
        out_shape = self.out_reg.shape
        out_shape_new = list(out_shape)

        if self.fire_type in (2, 3, 4, 7):
            out_shape_new[1] = np.ceil(out_shape_new[1] / 16).astype(int32_t) *
       16 elif self.fire_type == 5: out_shape_new[1] = np.ceil(out_shape_new[1]
       / 64).astype(int32_t) * 64

        out_reg = np.zeros(out_shape_new).astype('int32')
        out_reg[:out_shape[0], :out_shape[1]] = self.out_reg[:, :]

        self.out_reg = out_reg
         */
    int32_t shape_1 = para->neuron_num;
    if ((para->fire_type == 2) || (para->fire_type == 3) ||
        (para->fire_type == 4) || (para->fire_type == 7)) {
        shape_1 = ceil(double(shape_1) / 16.0) * 16;
    } else if (para->fire_type == 5) {
        shape_1 = ceil(double(shape_1) / 64.0) * 64;
    }
    auto p_new_out_reg = new_array<int32_t>(para->group_num, shape_1);
    int32_t **new_out_reg = (int32_t **)p_new_out_reg.get();

    for (int32_t i = 0; i < para->group_num; i++) {
        for (int32_t j = 0; j < para->neuron_num; j++) {
            new_out_reg[i][j] = out_reg[i * para->neuron_num + j];
        }
    }

    // self.seed = self.lfsr_vec[-1]
    //         para->seed=lfsr_vec[neuron_length-1];
    // V_new(self.V), Vtheta_updata(self.Vtheta), new_out_reg(self.out_reg),
    // ref_cnt(self.ref_cnt)

#if P08_OUTPUT_DEBUG
    fstream out_new_out_reg("c_out_new_out_reg.txt", ios::out);
    for (int32_t i = 0; i < para->group_num; i++) {
        for (int32_t j = 0; j < shape_1; j++) {
            out_new_out_reg << new_out_reg[i][j] << endl;
        }
    }
    out_new_out_reg.close();

#endif

    /*
         * if self.fire_type in (0, 1, 6):
            for i in range(out_shape[0]):
                for j in range(out_shape[1]):
                    _out_reg.append([self.out_reg[i][j]])
        elif self.fire_type in (2, 3, 4, 7):
            for i in range(out_shape[0]):
                for j in range(out_shape[1] // 4):
                    tmp = []
                    for k in range(4):
                        tmp.append(self.out_reg[i][j * 4 + k])
                    _out_reg.append(tmp)
        elif self.fire_type == 5:
            for i in range(out_shape[0]):
                for j in range(out_shape[1] // 16):
                    tmp = []
                    for k in range(8):
                        tmp.append(self.out_reg[i][j * 16 + k * 2] & 0x3)
                        tmp.append(self.out_reg[i][j * 16 + k * 2 + 1] & 0x3)
                    _out_reg.append(tmp)
        _Vresult.append(_out_reg)
         */
    shared_ptr<int32_t> p_out_reg;
    if ((para->fire_type == 1) || (para->fire_type == 0) ||
        (para->fire_type == 6)) {
        p_out_reg = new_array<int32_t>(para->group_num * shape_1);
        int32_t *p = p_out_reg.get();
        for (int32_t i = 0; i < para->group_num; ++i) {
            for (int32_t j = 0; j < shape_1; ++j) {
                p[i * shape_1 + j] = new_out_reg[i][j];
            }
        }
    } else if ((para->fire_type == 2) || (para->fire_type == 3) ||
               (para->fire_type == 4) || (para->fire_type == 7)) {
        p_out_reg = new_array<int32_t>(para->group_num * shape_1 / 4);
        int32_t *p = p_out_reg.get();
        for (int32_t i = 0; i < para->group_num; ++i) {
            for (int32_t j = 0; j < shape_1 / 4; ++j) {
                int32_t temp = 0;
                for (int32_t k = 0; k < 4; ++k) {
                    temp |= (new_out_reg[i][j * 4 + k] & 0xff) << (k * 8);
                }
                p[i * shape_1 / 4 + j] = temp;
            }
        }
    } else if (para->fire_type == 5) {
        p_out_reg = new_array<int32_t>(para->group_num * shape_1 / 16);
        int32_t *p = p_out_reg.get();
        for (int32_t i = 0; i < para->group_num; ++i) {
            for (int32_t j = 0; j < shape_1 / 16; ++j) {
                int32_t temp = 0;
                for (int32_t k = 0; k < 16; ++k) {
                    temp |= (new_out_reg[i][j * 16 + k] & 0b11) << (k * 2);
                }
                p[i * shape_1 / 16 + j] = temp;
            }
        }
    }

    /*
         * V_ref_cnt = (self.ref_cnt.astype('int32') << 28) | (self.V & (2 ** 28
       - 1)) # print(V_ref_cnt[0:8]) _V_ref_cnt = [] for i in
       range(self.neuron_length): _V_ref_cnt.append([V_ref_cnt[i]])
        _Vresult.append(_V_ref_cnt)
         */

    shared_ptr<int32_t> p_out_ref_cnt = new_array<int32_t>(neuron_length);
    int32_t *p_ref_cnt = p_out_ref_cnt.get();
    divider = (1 << 28) - 1;
    for (int32_t i = 0; i < neuron_length; ++i) {
        p_ref_cnt[i] = (ref_cnt[i] << 28) | (V_new[i] & divider);
    }

    /*
         *
        for i in range(self.Vtheta_len):
            _Vtheta.append([self.Vtheta[i]])
        _Vresult.append(_Vtheta)
         */
    shared_ptr<int32_t> p_out_Vtheta = new_array<int32_t>(para->Vtheta_len);
    int32_t *p_Vtheta = p_out_Vtheta.get();
    for (int32_t i = 0; i < para->Vtheta_len; ++i) {
        p_Vtheta[i] = Vtheta_updata[i];
    }

    /*
         * _para = []
        _para.append([self.seed])
        data = ((self.Vth0 & 0xffffff) << 8) | (self.Tw_cnt_new & 0xff)
        _para.append([data])
        data = ((self.Vth_beta & 0xfffff) << 12) | ((self.Vth_alpha & 0xff) <<
       4) | (self.Vth0 >> 24) & 0xf _para.append([data]) data = ((self.Vth_Incre
       & 0xffffff) << 8) | ((self.Vth_beta >> 20) & 0xff) _para.append([data])
        _Vresult.append(_para)
         */
    shared_ptr<int32_t> p_out_para = new_array<int32_t>(4);
    int32_t *p_para = p_out_para.get();
    p_para[0] = lfsr_vec[neuron_length - 1];
    p_para[1] = ((para->Vth0 & 0xffffff) << 8) | (Tw_cnt_new & 0xff);
    p_para[2] = ((para->Vth_beta & 0xfffff) << 12) |
                ((para->Vth_alpha & 0xff) << 4) | ((para->Vth0 >> 24) & 0xf);
    p_para[3] =
        ((para->Vth_Incre & 0xffffff) << 8) | ((para->Vth_beta >> 20) & 0xff);
    return {p_out_reg, p_out_ref_cnt, p_out_Vtheta, p_out_para};
}
