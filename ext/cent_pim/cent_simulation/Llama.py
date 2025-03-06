import math
import torch
import torch.nn.functional as F
from aim_sim import PIM
from TransformerBlock import TransformerBlock
from utils import compare, apply_rotary_emb, repeat_kv, RMSNorm

debug = True

class TransformerBlockLlama(TransformerBlock):
    """
    TransformerBlock Class inherits computate functionality from PIM class
    """
    def __init__(self, dic_model, args):
        super().__init__(dic_model, args)
        
    def precision_test(self):
        # Results are different in BFloat16 in 7B
        a = RMSNorm(self.x, self.SANorm)[0][0]
        b = self.wq[34]
        c = self.wq.T[:, 34:35]
        print(a)
        print(b)
        print(c)

        print(self.Vector_Vector_Mul(a, b, False))
        print((a*b).sum())
        print(torch.matmul(a, b))
        print(torch.matmul(a, c))

    def self_attention(self):
        bsz, seqlen, _ = self.x.shape

        RMSNorm_x = RMSNorm(self.x, self.SANorm)

        xq = F.linear(RMSNorm_x, self.wq)
        xk = F.linear(RMSNorm_x, self.wk)
        xv = F.linear(RMSNorm_x, self.wv)
        compare(xq[0][0], self.xq[0][0], "xq")
        compare(xk[0][0], self.xk[0][0], "xk")
        compare(xv[0][0], self.xv[0][0], "xv")
        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis)
        self.cache_k[:bsz, self.start_pos : self.start_pos + seqlen] = xk
        self.cache_v[:bsz, self.start_pos : self.start_pos + seqlen] = xv
        keys = self.cache_k[:bsz, : self.start_pos + seqlen]
        values = self.cache_v[:bsz, : self.start_pos + seqlen]
        if self.GQA:
            keys = repeat_kv(keys, self.n_repeat)
            values = repeat_kv(values, self.n_repeat)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2).transpose(2, 3)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys) / math.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1).type_as(xq)
        compare(scores[0][0], self.scores[0][0], "scores")

        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)
        compare(output[0][0], self.output[0][0], "output")
        sa = F.linear(output, self.wo)
        compare(sa[0][0], self.sa[0][0], "sa")
        sa = self.x + sa
        return sa
    
    def self_attention_aim(self):
        bsz, _, _ = self.x.shape
        seqlen = self.start_pos.item() + 1
        if self.model_parallel:
            FC_total_banks = self.total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = self.total_banks
            channels_required = self.channels_per_block
        channel_lst = [channel for channel in range(channels_required)]
        channel_multi_transformer_block_required = self.num_channels // channels_required * channels_required
        channel_lst_multi_transformer_block = [channel for channel in range(channel_multi_transformer_block_required)]

        # AiM MAC BK x BK
        if self.pim_compute:
            x_pow_sum = 0
            op_size = (self.dic_shape["x_neighbor_bank"][0] - 1) // self.burst_length + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.WR_BIAS(0, channel, channels_required, 0, [0 for bank in range(self.num_banks)], op_trace)
                self.MAC_BK_BK(0, channel, channels_required, self.x_row_index, 0, 0, op_size, op_trace)
                mac_lst = self.RD_MAC(0, channel, channels_required, 0, op_trace)
                x_pow_sum += sum(mac_lst)    # CXL ports
            compare(x_pow_sum, self.x.pow(2).sum(), "x_pow_sum")
        else:
            x_load = self.load_from_DRAM_multi_channel(self.x.shape, self.x_row_index, "vector_neighbor_bank_0", self.dic_shape["x_neighbor_bank"][0], False)
            x_copy_load = self.load_from_DRAM_multi_channel(self.x.shape, self.x_row_index, "vector_neighbor_bank_1", self.dic_shape["x_neighbor_bank"][0], False)
            x_pow_sum = self.Vector_Vector_Mul(x_load[0][0], x_copy_load[0][0], False)

        # CXL Ports     x_copy -> norm_tensor
        norm = torch.rsqrt(x_pow_sum / self.dim + 1e-5)
        norm_tensor = torch.full(self.x.shape, norm)
        self.store_to_DRAM_multi_channel(norm_tensor[0][0], self.x_copy_row_index, "vector_bank_group_1", self.trace_norm)
        self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
        
        # AiM EWMUL     Load x and norm_tensor      norm_tensor -> norm_x -> RMSNorm_x_aim
        if self.pim_compute:
            op_size = (self.dic_shape["x_bank_group"][0] - 1) // self.burst_length + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.x_copy_row_index, 0, op_size, op_trace)
        else:
            norm_tensor_load = self.load_from_DRAM_multi_channel(self.x.shape, self.x_copy_row_index, "vector_bank_group_1", self.dic_shape["x_bank_group"][0], False)
            x_load = self.load_from_DRAM_multi_channel(self.x.shape, self.x_copy_row_index, "vector_bank_group_0", self.dic_shape["x_bank_group"][0], False)
            norm_x = self.Vector_Vector_EWMUL(x_load, norm_tensor_load)
            self.store_to_DRAM_multi_channel(norm_x[0][0], self.x_copy_row_index, "vector_bank_group_2", False)

        # AiM EWMUL     Copy norm_x to SANorm row
        if self.pim_compute:
            op_size = (self.dic_shape["x_bank_group"][0] - 1) // self.burst_length + 1
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_norm
                    self.COPY_BK_GB(0, channel, channels_required, bank, self.x_copy_row_index, 0, op_size, op_trace)
                    self.COPY_GB_BK(0, channel, channels_required, bank-1, self.SANorm_row_index, 0, op_size, op_trace)
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.SANorm_row_index, 0, op_size, op_trace)
        else:
            self.store_to_DRAM_multi_channel(norm_x[0][0], self.SANorm_row_index, "vector_bank_group_1", False)
            norm_x_load = self.load_from_DRAM_multi_channel(self.x.shape, self.SANorm_row_index, "vector_bank_group_1", self.dic_shape["x_bank_group"][0], False)
            SANorm_load = self.load_from_DRAM_multi_channel(self.x.shape, self.SANorm_row_index, "vector_bank_group_0", self.dic_shape["SANorm_bank_group"][0], False)
            RMSNorm_x_aim = self.Vector_Vector_EWMUL(norm_x_load, SANorm_load)
            self.store_to_DRAM_multi_channel(RMSNorm_x_aim[0][0], self.SANorm_row_index, "vector_bank_group_2", False)
        
        # Broadcast the scattered RMSNorm_x_aim results to all channels
        RMSNorm_x_aim = self.load_from_DRAM_multi_channel(self.x.shape, self.SANorm_row_index, "vector_bank_group_2", self.dic_shape["x_bank_group"][0], self.trace_norm)
        compare(RMSNorm_x_aim[0][0], RMSNorm(self.x[0][0], self.SANorm), "RMSNorm_x_aim")
        
        # AiM MAC BK x GB
        if self.pim_compute:
        # if False:
            xq_aim = self.Vector_Matrix_Mul_weight_pim(RMSNorm_x_aim[0][0], self.wq_row_index, self.dim, self.wq.shape[0], FC_total_banks, self.trace_fc_kqvo, "breakdown_sa_weight").reshape(bsz, 1, -1)
            xk_aim = self.Vector_Matrix_Mul_weight_pim(RMSNorm_x_aim[0][0], self.wk_row_index, self.dim, self.wk.shape[0], FC_total_banks, self.trace_fc_kqvo, "breakdown_sa_weight").reshape(bsz, 1, -1)
            xv_aim = self.Vector_Matrix_Mul_weight_pim(RMSNorm_x_aim[0][0], self.wv_row_index, self.dim, self.wv.shape[0], FC_total_banks, self.trace_fc_kqvo, "breakdown_sa_weight").reshape(bsz, 1, -1)
        else:
            wq_aim = self.load_from_DRAM_multi_channel(self.wq.shape, self.wq_row_index, self.mode["weights"], self.dic_shape["wq"][0], False)
            wk_aim = self.load_from_DRAM_multi_channel(self.wk.shape, self.wk_row_index, self.mode["weights"], self.dic_shape["wk"][0], False)
            wv_aim = self.load_from_DRAM_multi_channel(self.wv.shape, self.wv_row_index, self.mode["weights"], self.dic_shape["wv"][0], False)
            xq_aim = self.Vector_Matrix_Mul_multithreads(RMSNorm_x_aim[0][0], wq_aim.T).reshape(bsz, 1, -1)
            xk_aim = self.Vector_Matrix_Mul_multithreads(RMSNorm_x_aim[0][0], wk_aim.T).reshape(bsz, 1, -1)
            xv_aim = self.Vector_Matrix_Mul_multithreads(RMSNorm_x_aim[0][0], wv_aim.T).reshape(bsz, 1, -1)
            # xq_aim = torch.tensor(self.Vector_Matrix_Mul(RMSNorm_x_aim[0][0], wq_aim.T)).reshape(bsz, 1, -1)
            # xk_aim = torch.tensor(self.Vector_Matrix_Mul(RMSNorm_x_aim[0][0], wk_aim.T)).reshape(bsz, 1, -1)
            # xv_aim = torch.tensor(self.Vector_Matrix_Mul(RMSNorm_x_aim[0][0], wv_aim.T)).reshape(bsz, 1, -1)
        compare(xq_aim[0][0], self.xq[0][0], "Vector_Matrix_Mul xq")
        compare(xk_aim[0][0], self.xk[0][0], "Vector_Matrix_Mul xk")
        compare(xv_aim[0][0], self.xv[0][0], "Vector_Matrix_Mul xv")

        # CXL Ports     rotary embedding
        xq_aim = xq_aim.reshape(bsz, 1, self.n_heads, self.head_dim)
        xk_aim = xk_aim.reshape(bsz, 1, self.n_kv_heads, self.head_dim)
        xv_aim = xv_aim.reshape(bsz, 1, self.n_kv_heads, self.head_dim)

        xq_aim, xk_aim = apply_rotary_emb(xq_aim, xk_aim, self.freqs_cis)

        if self.trace_fc_kqvo:
            input_vector_EWMUL_length = (self.dim - 1) // (self.total_banks // 4) + 1
            input_vector_EWMUL_utilized_banks = (self.dim - 1) // input_vector_EWMUL_length + 1
            # Store re-mapped xq/xk for EWMUL
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)
            # Rotary embedding
            self.EWMUL_only_trace(channel_lst_multi_transformer_block, self.xq_row_index, self.dim // self.burst_length)
            self.EWMUL_only_trace(channel_lst_multi_transformer_block, self.xk_row_index, self.dim // self.n_repeat // self.burst_length)
            # Load rotary embedding results
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)

        self.dic_shape["xq"] = self.store_to_DRAM_multi_channel(xq_aim.reshape(-1), self.xq_row_index, self.mode["vector"], False)

        # CXL Ports     Store xq
        if self.pim_compute:
            self.broadcast_store_query(channels_required, self.xq_row_index, xq_aim.reshape(-1), False)
            xq_aim_loaded = {}
            self.broadcast_load_query(xq_aim_loaded, channels_required, self.xq_row_index)
            print()
            for channel in channel_lst:
                compare(xq_aim.reshape(-1), xq_aim_loaded[channel], "xq_aim_loaded channel "+str(channel))
        else:
            xq_aim_load = self.load_from_DRAM_multi_channel(self.xq.shape, self.xq_row_index, self.mode["vector"], self.dic_shape["xq"][0], False)
            xq_aim_load = xq_aim_load.reshape(bsz, 1, self.n_heads, self.head_dim)
            compare(xq_aim_load[0][0], xq_aim[0][0], "xq_aim_load")

        # CXL Ports     Store xk xv
        if self.pim_compute:
            # cache k and v in weights have reserved the positions for xk and xv, for each processed token, we need to store the new xk/xv to the correct position
            seq = seqlen - 1
            dimm_index, channel_index, bank_index = self.bank_index(seq % self.FC_total_banks)
            xk_data = xk_aim.reshape(-1)
            rows = self.head_dim * self.n_kv_heads // self.DRAM_column
            for row in range(rows):
                data_row = xk_data[row * self.DRAM_column : (row + 1) * self.DRAM_column]
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.DRAM_column // self.burst_length
                self.store_to_DRAM_single_bank(dimm_index, channel_index, bank_index, self.cache_k_row_index + seq // self.FC_total_banks * rows + row, 0, self.DRAM_column, data_row, self.trace_attention)

            if self.intra_device_attention:   # old V cache mapping, which is not friendly for long context scenario. E.g. seqlen=32k, it stores 32 rows data (1k per row) in each bank, and utilizes 128 banks (8 channels) per head. In Llama2-70B with GQA, only 2 devices (16x32x2=1k banks) can be used. 
                xv_data = xv_aim.transpose(1, 2).transpose(2, 3)
                num_rows_per_seq = (seq - 1) // self.DRAM_column + 1
                rows_per_dim = self.max_seq_len // self.DRAM_column
                num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
                dim_iteration = self.head_dim // self.num_banks
                for head_index_per_bank in range(num_heads_per_bank):     # each head is distributed into all banks in a channel, each bank contains left_banks heads
                    row_current_head = self.cache_v_row_index + (rows_per_dim * dim_iteration) * head_index_per_bank
                    for dim_iter in range(dim_iteration):   # each head has dim 128, but distributed to 16 banks, so has 8 iterations in each bank
                        for channel in channel_lst:
                            head = channel * num_heads_per_bank + head_index_per_bank
                            if head > self.n_kv_heads - 1:
                                break
                            if self.trace_attention:
                                self.WR_ABK_only_trace(channel, row_current_head + dim_iter * rows_per_dim + num_rows_per_seq - 1, 1)
                            # self.store_to_DRAM_all_banks(dim_iter, channel, row_current_head, seq, head, xv_data, num_rows_per_seq, rows_per_dim)
                            for bank in range(self.num_banks):
                                dim = dim_iter * self.num_banks + bank
                                row_offset = num_rows_per_seq - 1
                                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + 1
                                self.store_to_DRAM_single_bank(0, channel, bank, row_current_head + dim_iter * rows_per_dim + row_offset, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)
            else:
                xv_data = xv_aim.transpose(1, 2).transpose(2, 3)
                channels_required_all_devices = self.FC_total_banks // self.num_banks
                # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
                # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
                # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
                # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                    # seqlen = 32k, head_dim = 128
                banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 32, 256, 2k
                channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 2,  16,  128
                devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,   4
                # iteration along the head dimension
                dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 4,  1,   1
                # iteration along the sequence dimension or rows per sequence
                rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  2,   16
                seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 16,  2
                rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32,  32
                channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
                for channel in range(channels_required_all_devices):
                    if banks_per_head < self.num_banks:
                        raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                    head = channel // (banks_per_head // self.num_banks)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            row_offset = rows_per_seq - 1
                            if self.trace_attention and channel < self.num_channels:
                                self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset * dim_iterations + dim_iter, 1)
                            for bank in range(self.num_banks):
                                dim = dim_iter * banks_per_head + (channel % channels_per_head) * self.num_banks + bank
                                self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, self.cache_v_row_index + row_offset * dim_iterations + dim_iter, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)
                    else:
                        # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                        # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                        # each head is mapped on multiple devices
                        # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                        for bank in range(self.num_banks):
                            dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                            if (channel % channels_per_head) // channels_per_row_offset == rows_per_seq - 1:
                                row_offset = rows_per_seq - 1
                                if self.trace_attention and channel < self.num_channels:
                                    self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset // rows_per_seq_iteration, 1)
                                for bank in range(self.num_banks):
                                    dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                                    self.store_to_DRAM_single_bank(channel // self.num_channels, channel % self.num_channels, bank, self.cache_v_row_index + row_offset // rows_per_seq_iteration, seq % self.DRAM_column, 1, xv_data[0][head][dim], False)

        else:
            cache_v_size = torch.Size([bsz, self.n_kv_heads, self.head_dim, -1])
            cache_k = self.load_from_DRAM_multi_channel(self.cache_k.shape, self.cache_k_row_index, self.mode["cache_k"], self.cache_k.shape[1], False)
            cache_v = self.load_from_DRAM_multi_channel(cache_v_size, self.cache_v_row_index, self.mode["cache_v"], self.cache_k.shape[1], False).transpose(2, 3).transpose(1, 2).reshape(bsz, -1, self.n_kv_heads, self.head_dim)
            compare(cache_k, self.cache_k, "cache v old")
            compare(cache_v, self.cache_v, "cache k old")
            cache_k[:bsz, self.start_pos : self.start_pos + 1] = xk_aim
            cache_v[:bsz, self.start_pos : self.start_pos + 1] = xv_aim

            keys_aim = cache_k[:bsz, : self.start_pos + 1]
            values_aim = cache_v[:bsz, : self.start_pos + 1]
            if self.GQA:
                keys_aim = repeat_kv(keys_aim, self.n_repeat)
                values_aim = repeat_kv(values_aim, self.n_repeat)
            xq_aim_load = xq_aim_load.transpose(1, 2)
            keys_aim = keys_aim.transpose(1, 2).transpose(2, 3)
            values_aim = values_aim.transpose(1, 2)

        # AiM MAC BK x GB
        if self.pim_compute:
            scores_aim = self.Vector_Matrix_Mul_score_pim(self.xq_row_index, self.cache_k_row_index, self.trace_attention, "breakdown_sa_score")

            if debug:
                self.cache_k[:bsz, self.start_pos : self.start_pos + seqlen] = xk_aim
                self.cache_v[:bsz, self.start_pos : self.start_pos + seqlen] = xv_aim
                keys = self.cache_k[:bsz, : self.start_pos + seqlen]
                values = self.cache_v[:bsz, : self.start_pos + seqlen]
                if self.GQA:
                    keys = repeat_kv(keys, self.n_repeat)
                    values = repeat_kv(values, self.n_repeat)
                xq = xq_aim.transpose(1, 2)
                keys = keys.transpose(1, 2).transpose(2, 3)
                compare(scores_aim, torch.matmul(xq, keys), "Vector_Matrix_Mul score")
        else:
            scores_aim = []
            for i in range(self.n_heads):
                scores_aim.append(self.Vector_Matrix_Mul(xq_aim_load[0][i][0], keys_aim[0][i], False))
            scores_aim = torch.tensor(scores_aim).reshape(bsz, self.n_heads, 1, -1)

        # CXL Ports
        head_dim_reciprocal = torch.full(scores_aim.shape, 1 / math.sqrt(self.head_dim))
        self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_0", self.trace_attention)
        self.store_to_DRAM_multi_channel(head_dim_reciprocal.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_1", self.trace_attention)
        for channel in channel_lst:
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average

        # AiM EWMUL
        if self.pim_compute:
            rows_per_score = (seqlen - 1) // self.DRAM_column + 1
            num_scores_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_attention
                for score_index in range(num_scores_per_bank):
                    for row in range(rows_per_score):
                        if row == rows_per_score - 1:
                            offset = seqlen - row * self.DRAM_column
                        else:
                            offset = self.DRAM_column
                        self.EWMUL(0, channel, channels_required, self.scores_row_index + score_index * rows_per_score + row, 0, (offset - 1) // self.burst_length + 1, op_trace)
        else:
            scores_aim_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_0", seqlen, False)
            head_dim_reciprocal_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_1", seqlen, False)
            scores_aim = self.Vector_Vector_EWMUL(scores_aim_load, head_dim_reciprocal_load)
            self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_2", False)

        # CXL Ports
        scores_aim = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_2", seqlen, self.trace_attention)
        if self.pim_compute and debug:
            scores = torch.matmul(xq, keys) / math.sqrt(self.head_dim)
            compare(scores_aim, scores, "Vector_Matrix_Mul score / head_dim")

        scores_exp = torch.exp(scores_aim)
        scores_exp_sum_reciprocal = 1 / torch.sum(scores_exp, dim=-1, keepdim=True)
        scores_exp_sum_reciprocal = torch.cat([scores_exp_sum_reciprocal for i in range(scores_exp.shape[-1])], dim=-1)
        self.store_to_DRAM_multi_channel(scores_exp.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_0", self.trace_attention)
        self.store_to_DRAM_multi_channel(scores_exp_sum_reciprocal.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_1", self.trace_attention)
        for channel in range(channels_required):
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average
            self.time["WR_SBK"] += (self.timing_constant["WR_SBK"] + 4096 // self.burst_length) * 16 // 2    # 16 rows for 4k seq length, but use 16 // 2 for average

        # AiM EWMUL
        if self.pim_compute:
            rows_per_score = (seqlen - 1) // self.DRAM_column + 1
            num_scores_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_attention
                for score_index in range(num_scores_per_bank):
                    for row in range(rows_per_score):
                        if row == rows_per_score - 1:
                            offset = seqlen - row * self.DRAM_column
                        else:
                            offset = self.DRAM_column
                        self.EWMUL(0, channel, channels_required, self.scores_row_index + score_index * rows_per_score + row, 0, (offset - 1) // self.burst_length + 1, op_trace)
            scores_aim = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_2", seqlen, self.trace_attention)
        else:
            scores_exp_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_0", seqlen, False)
            scores_exp_sum_reciprocal_load = self.load_from_DRAM_multi_channel(self.scores.shape, self.scores_row_index, "scores_bank_group_1", seqlen, False)
            scores_aim = self.Vector_Vector_EWMUL(scores_exp_load, scores_exp_sum_reciprocal_load)
            self.store_to_DRAM_multi_channel(scores_aim.reshape(self.n_heads, -1), self.scores_row_index, "scores_bank_group_2", False)
        compare(scores_aim, self.scores, "SoftMax scores")

        # AiM MAC BK x GB
        if self.pim_compute:
            output_aim = self.Vector_Matrix_Mul_output_pim(scores_aim, self.cache_v_row_index, self.trace_attention, "breakdown_sa_output").reshape(bsz, 1, -1)
        else:
            output_aim = []
            for i in range(self.n_heads):
                output_aim.append(self.Vector_Matrix_Mul(scores_aim[0][i][0], values_aim[0][i], False))
            output_aim = torch.tensor(output_aim).reshape(bsz, 1, -1)
        compare(output_aim[0][0], self.output[0][0], "Vector_Matrix_Mul output")

        # CXL Ports
        self.dic_shape["output"] = self.store_to_DRAM_multi_channel(output_aim[0][0], self.output_row_index, self.mode["vector"], False)

        # AiM MAC BK x GB
        if self.pim_compute:
            sa_aim = self.Vector_Matrix_Mul_weight_pim(output_aim[0][0], self.wo_row_index, self.dim, self.wo.shape[0], FC_total_banks, self.trace_fc_kqvo, "breakdown_sa_weight").reshape(bsz, 1, -1)
        else:
            wo_aim = self.load_from_DRAM_multi_channel(self.wo.shape, self.wo_row_index, self.mode["weights"], self.dic_shape["wo"][0], False)
            sa_aim = self.Vector_Matrix_Mul_multithreads(output_aim[0][0], wo_aim.T).reshape(bsz, 1, -1)
        self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_row_index, "vector_bank_group_0", False)
        compare(sa_aim[0][0], self.sa[0][0], "Vector_Matrix_Mul sa")

        # CXL Ports
        sa_aim_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
        x_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
        sa_aim = self.Vector_Vector_EWADD(x_load, sa_aim_load)
        self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_row_index, "vector_bank_group_2", False)

        return sa_aim
    
    def FFN(self, sa):
        compare(sa[0][0], self.h[0][0], "h")
        RMSNorm_sa = RMSNorm(sa, self.FFNNorm)
        x1 = F.linear(RMSNorm_sa, self.w1)
        x3 = F.linear(RMSNorm_sa, self.w3)
        ffn = F.linear(F.silu(x1) * x3, self.w2)
        compare(ffn[0][0], self.ffn[0][0], "ffn")
        out = sa + ffn
        return out
    
    def FFN_aim(self, sa_aim):
        bsz, _, _ = self.sa.shape
        if self.model_parallel:
            FC_total_banks = self.total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = self.total_banks
            channels_required = self.channels_per_block
        channel_lst = [channel for channel in range(channels_required)]

        # AiM MAC BK x BK
        if self.pim_compute:
            self.dic_shape["sa_neighbor_bank"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_neighbor_bank_0", self.trace_norm)
            self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_neighbor_bank_1", self.trace_norm)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            sa_pow_sum = 0
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.WR_BIAS(0, channel, channels_required, 0, [0 for bank in range(self.num_banks)], op_trace)
                op_size = (self.dic_shape["sa_neighbor_bank"][0] - 1) // self.burst_length + 1
                self.MAC_BK_BK(0, channel, channels_required, self.sa_copy_row_index, 0, 0, op_size, op_trace)
                sa_pow_sum += sum(self.RD_MAC(0, channel, channels_required, 0, op_trace))    # CXL ports
        else:
            self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_0", False)
            self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_1", False)
            sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
            sa_copy_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            sa_pow_sum = self.Vector_Vector_Mul(sa_load[0][0], sa_copy_load[0][0], False)

        # CXL Ports
        compare(sa_pow_sum, sa_aim.pow(2).sum(), "sa pow")
        norm = torch.rsqrt(sa_pow_sum / self.dim + 1e-5)
        norm_tensor = torch.full(sa_aim.shape, norm)
        self.store_to_DRAM_multi_channel(norm_tensor[0][0], self.sa_copy_row_index, "vector_bank_group_1", self.trace_norm)
        self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length

        # AiM EWMUL
        if self.pim_compute:
            self.dic_shape["sa_bank_group"] = self.store_to_DRAM_multi_channel(sa_aim[0][0], self.sa_copy_row_index, "vector_bank_group_0", self.trace_norm)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.x.shape[-1] // self.burst_length
            op_size = (self.dic_shape["sa_bank_group"][0] - 1) // self.burst_length + 1
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.sa_copy_row_index, 0, op_size, op_trace)
        else:
            norm_tensor_load = self.load_from_DRAM_multi_channel(self.x.shape, self.sa_copy_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.sa_copy_row_index, "vector_bank_group_0", self.dic_shape["sa_bank_group"][0], False)
            norm_sa = self.Vector_Vector_EWMUL(sa_load, norm_tensor_load)
            self.store_to_DRAM_multi_channel(norm_sa[0][0], self.FFNNorm_row_index, "vector_bank_group_2", False)

        # AiM EWMUL
        if self.pim_compute:
            op_size = (self.dic_shape["sa_bank_group"][0] - 1) // self.burst_length + 1
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_norm
                    self.COPY_BK_GB(0, channel, channels_required, bank, self.sa_copy_row_index, 0, op_size, op_trace)
                    self.COPY_GB_BK(0, channel, channels_required, bank-1, self.FFNNorm_row_index, 0, op_size, op_trace)
            for channel in channel_lst:
                op_trace = channel == 0 and self.trace_norm
                self.EWMUL(0, channel, channels_required, self.FFNNorm_row_index, 0, op_size, op_trace)
            FFNNorm_sa_aim = self.load_from_DRAM_multi_channel(self.x.shape, self.FFNNorm_row_index, "vector_bank_group_2", self.dic_shape["FFNNorm"][0], self.trace_norm)
        else:
            self.store_to_DRAM_multi_channel(norm_sa[0][0], self.FFNNorm_row_index, "vector_bank_group_1", False)
            norm_sa_load = self.load_from_DRAM_multi_channel(self.x.shape, self.FFNNorm_row_index, "vector_bank_group_1", self.dic_shape["sa_bank_group"][0], False)
            FFNNorm_load = self.load_from_DRAM_multi_channel(self.FFNNorm.shape, self.FFNNorm_row_index, "vector_bank_group_0", self.dic_shape["FFNNorm"][0], False)
            FFNNorm_sa_aim = self.Vector_Vector_EWMUL(norm_sa_load, FFNNorm_load)
            self.store_to_DRAM_multi_channel(FFNNorm_sa_aim[0][0], self.FFNNorm_row_index, "vector_bank_group_2", False)

        bsz, _, _ = FFNNorm_sa_aim.shape
        compare(FFNNorm_sa_aim[0][0], RMSNorm(sa_aim[0][0], self.FFNNorm), "FFNNorm_sa_aim")

        # AiM MAC BK x GB
        if self.pim_compute:
            x1_aim, x1_sigmoid_aim = self.Vector_Matrix_Mul_weight_af_pim(FFNNorm_sa_aim[0][0], self.w1_row_index, self.dim, self.w1.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight")
            x1_aim = x1_aim[:self.w1.shape[0]].reshape(bsz, 1, -1)
            x1_sigmoid_aim = x1_sigmoid_aim[:self.w1.shape[0]].reshape(bsz, 1, -1)
            x3_aim = self.Vector_Matrix_Mul_weight_pim(FFNNorm_sa_aim[0][0], self.w3_row_index, self.dim, self.w1.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight")[:self.w3.shape[0]].reshape(bsz, 1, -1)
        else:
            w1_aim = self.load_from_DRAM_multi_channel(self.w1.shape, self.w1_row_index, self.mode["weights"], self.dic_shape["w1"][0], False)
            w3_aim = self.load_from_DRAM_multi_channel(self.w3.shape, self.w3_row_index, self.mode["weights"], self.dic_shape["w3"][0], False)
            x1_aim = self.Vector_Matrix_Mul_multithreads(FFNNorm_sa_aim[0][0], w1_aim.T).reshape(bsz, 1, -1)
            x3_aim = self.Vector_Matrix_Mul_multithreads(FFNNorm_sa_aim[0][0], w3_aim.T).reshape(bsz, 1, -1)
            self.dic_shape["x1"] = self.store_to_DRAM_multi_channel(x1_aim[0][0], self.x1_row_index, self.mode["vector"], False)
            self.dic_shape["x3"] = self.store_to_DRAM_multi_channel(x3_aim[0][0], self.x3_row_index, self.mode["vector"], False)
        compare(x1_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w1.T), "x1")
        compare(x3_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w3.T), "x3")

        # AiM AF EWMUL
        x1_sigmoid = torch.sigmoid(x1_aim)
        if self.pim_compute:
            # compare(x1_sigmoid_aim, torch.sigmoid(x1_aim), "x1 sigmoid")
            iteration_required = x1_aim.shape[-1] > self.channels_per_block * (self.num_banks // 4) * self.DRAM_column
            if iteration_required:
                iteration_0 = self.total_banks // 4 * 1024
                self.dic_shape["x1_bank_group_0"] = self.store_to_DRAM_multi_channel(x1_aim[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_0", self.trace_activation)
                self.dic_shape["x1_bank_group_1"] = self.store_to_DRAM_multi_channel(x1_aim[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_1", self.trace_activation)
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_1", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                x1_silu_0 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, iteration_0]), self.x1_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_0"][0], False)
                x1_silu_1 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, self.w1.shape[0] - iteration_0]), self.x1_sigmoid_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_1"][0], False)
                x1_silu = torch.cat((x1_silu_0, x1_silu_1), dim=2)
            else:
                self.dic_shape["x1_bank_group"] = self.store_to_DRAM_multi_channel(x1_aim[0][0], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.store_to_DRAM_multi_channel(x1_sigmoid[0][0], self.x1_sigmoid_row_index, "vector_bank_group_1", self.trace_activation)
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                x1_silu = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, "vector_bank_group_2", self.dic_shape["x1_bank_group"][0], False)
            compare(x1_silu[0][0], (x1_aim * x1_sigmoid)[0][0], "x1_silu")
        else:
            compare(x1_aim[0][0], torch.matmul(FFNNorm_sa_aim[0][0], self.w1.T), "x1")
            self.dic_shape["x1_sigmoid"] = self.store_to_DRAM_multi_channel(x1_sigmoid[0][0], self.x1_sigmoid_row_index, self.mode["vector"], False)
            x1_aim_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_sigmoid_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_silu = self.Vector_Vector_EWMUL(x1_aim_load, x1_sigmoid_load)
            self.store_to_DRAM_multi_channel(x1_silu[0][0], self.x1_sigmoid_row_index, self.mode["vector"], False)

        # AiM EWMUL
        if self.pim_compute:
            for bank in [2, 6, 10, 14]:
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    if iteration_required:
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                    else:
                        self.COPY_BK_GB(0, channel, channels_required, bank, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                        self.COPY_GB_BK(0, channel, channels_required, bank-1, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
            if iteration_required:
                self.store_to_DRAM_multi_channel(x3_aim[0][0][:iteration_0], self.x1_row_index, "vector_bank_group_0", self.trace_activation)
                self.store_to_DRAM_multi_channel(x3_aim[0][0][iteration_0:], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * 2 + self.w1.shape[0] // self.burst_length
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_row_index, 0, (self.dic_shape["x1_bank_group_0"][0] - 1) // self.burst_length + 1, op_trace)
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group_1"][0] - 1) // self.burst_length + 1, op_trace)
                ffn_vector_0 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, iteration_0]), self.x1_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_0"][0], self.trace_activation)
                ffn_vector_1 = self.load_from_DRAM_multi_channel(torch.Size([1, 1, self.w1.shape[0] - iteration_0]), self.x1_sigmoid_row_index, "ffn_bank_group_2", self.dic_shape["x1_bank_group_1"][0], self.trace_activation)
                ffn_vector = torch.cat((ffn_vector_0, ffn_vector_1), dim=2)
            else:
                self.store_to_DRAM_multi_channel(x3_aim[0][0], self.x1_sigmoid_row_index, "vector_bank_group_0", self.trace_activation)
                for channel in channel_lst:
                    op_trace = channel == 0 and self.trace_activation
                    self.EWMUL(0, channel, channels_required, self.x1_sigmoid_row_index, 0, (self.dic_shape["x1_bank_group"][0] - 1) // self.burst_length + 1, op_trace)
                ffn_vector = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, "vector_bank_group_2", self.dic_shape["x1_bank_group"][0], self.trace_activation)
        else:
            x3_aim_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x3_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            x1_silu_load = self.load_from_DRAM_multi_channel(x1_aim.shape, self.x1_sigmoid_row_index, self.mode["vector"], self.dic_shape["x1"][0], False)
            ffn_vector = self.Vector_Vector_EWMUL(x1_silu_load, x3_aim_load)
        compare(ffn_vector[0][0], (F.silu(x1_aim) * x3_aim)[0][0], "ffn_vector")

        # AiM MAC BK x GB
        if self.pim_compute:
            ffn_aim = self.Vector_Matrix_Mul_weight_pim(ffn_vector[0][0], self.w2_row_index, self.w1.shape[0], self.w2.shape[0], FC_total_banks, self.trace_fc_ffn, "breakdown_ffn_weight").reshape(bsz, 1, -1)
        else:
            w2_aim = self.load_from_DRAM_multi_channel(self.w2.shape, self.w2_row_index, self.mode["weights"], self.dic_shape["w2"][0], False)
            ffn_aim = self.Vector_Matrix_Mul_multithreads(ffn_vector[0][0], w2_aim.T).reshape(bsz, 1, -1)
        self.dic_shape["ffn_bank_group"] = self.store_to_DRAM_multi_channel(ffn_aim[0][0], self.ffn_row_index, "vector_bank_group_1", False)
        compare(ffn_aim[0][0], self.ffn[0][0], "Vector_Matrix_Mul ffn")

        # AiM EWADD
        self.store_to_DRAM_multi_channel(sa_aim[0][0], self.ffn_row_index, "vector_bank_group_0", False)
        sa_load = self.load_from_DRAM_multi_channel(self.sa.shape, self.ffn_row_index, "vector_bank_group_0", self.dic_shape["ffn_bank_group"][0], False)
        ffn_load = self.load_from_DRAM_multi_channel(self.ffn.shape, self.ffn_row_index, "vector_bank_group_1", self.dic_shape["ffn_bank_group"][0], False)
        out_aim = self.Vector_Vector_EWADD(sa_load, ffn_load)
        self.dic_shape["out_bank_group"] = self.store_to_DRAM_multi_channel(out_aim[0][0], self.ffn_row_index, "vector_bank_group_2", False)

        return out_aim
    
    def trace_only(self):
        bsz, _, _ = self.x.shape
        seqlen = self.seqlen
        total_banks = self.total_banks
        if self.model_parallel:
            FC_total_banks = total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = total_banks
            channels_required = self.channels_per_block
        channel_multi_transformer_block_required = self.num_channels // channels_required * channels_required
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]
        num_transformer_blocks_per_device = max(self.num_channels // channels_required, 1)

        input_vector_neighbor_bank_length = (self.dim - 1) // (self.total_banks // 2) + 1
        input_vector_neighbor_bank_utilized_banks = (self.dim - 1) // input_vector_neighbor_bank_length + 1
        if self.trace_norm:
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 0, self.x_row_index, input_vector_neighbor_bank_length)
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 1, self.x_row_index, input_vector_neighbor_bank_length)

        # RMSNorm   x.pow   MAC_ABK
        input_vector_MAB_BK_BK_length = (self.dim - 1) // (total_banks // 2) + 1
        if self.trace_norm:
            self.WR_BIAS_only_trace(channel_lst)
            self.MAC_ABK_only_trace(channel_lst, self.x_row_index, (input_vector_MAB_BK_BK_length - 1) // self.burst_length + 1, "breakdown_sa_pow")
            self.RD_MAC_only_trace(channel_lst)

        # CXL Port  
        # Reduction of dim // 16 intermidiate sum read from MAC
        # Broadcast a scalar to vector and store it for EWMUL
        input_vector_EWMUL_length = (self.dim - 1) // (total_banks // 4) + 1
        input_vector_EWMUL_utilized_banks = (self.dim - 1) // input_vector_EWMUL_length + 1
        if self.trace_norm:
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.x_copy_row_index, input_vector_EWMUL_length)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.x_copy_row_index, input_vector_EWMUL_length)

            # RMSNorm   EWMUL
            self.EWMUL_only_trace(channel_lst, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            for bank in range(self.num_banks):
                bank_group_index = 2
                if bank % 4 == bank_group_index:
                    self.COPY_BK_GB_only_trace(channel_lst, bank, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
                    self.COPY_GB_BK_only_trace(channel_lst, bank-1, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
            self.EWMUL_only_trace(channel_lst, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            # Read RMSNorm result vector to GPR
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim // self.burst_length
            self.load_from_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.SANorm_row_index, input_vector_EWMUL_length)
            self.SYNC_only_trace()

        # K/Q/V GEMV
        if self.trace_fc_kqvo:
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wq_row_index, self.dim, self.head_dim * self.n_heads, FC_total_banks, "breakdown_sa_weight")
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wk_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wv_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")

            # CXL Port
            # Store re-mapped xq/xk for EWMUL
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)
            # Rotary embedding
            self.EWMUL_only_trace(channel_lst, self.xq_row_index, self.dim // self.burst_length)
            self.EWMUL_only_trace(channel_lst, self.xk_row_index, self.dim // self.n_repeat // self.burst_length)
            # Load rotary embedding results
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xq_row_index, input_vector_EWMUL_length * 2)
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim * 2 // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.xk_row_index, input_vector_EWMUL_length // self.n_repeat * 2)

        if self.trace_attention:
            # Store xk
            seq = seqlen - 1
            dimm_index, channel_index, bank_index = self.bank_index(seq % self.FC_total_banks)
            rows = self.head_dim * self.n_kv_heads // self.DRAM_column
            for row in range(rows):
                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.DRAM_column // self.burst_length
                for tb in range(num_transformer_blocks_per_device):
                    self.W_MEM_only_trace(channel_index + tb * channels_required, bank_index, self.cache_k_row_index + seq // self.FC_total_banks * rows + row, self.DRAM_column)
            # Store xv
            if self.intra_device_attention:
                num_rows_per_seq = (seq - 1) // self.DRAM_column + 1
                row_offset = num_rows_per_seq - 1
                rows_per_dim = self.max_seq_len // self.DRAM_column
                num_heads_per_bank = (self.n_kv_heads - 1) // self.channels_per_block + 1
                dim_iteration = self.head_dim // self.num_banks
                for head_index_per_bank in range(num_heads_per_bank):
                    row_current_head = self.cache_v_row_index + (rows_per_dim * dim_iteration) * head_index_per_bank
                    for dim_iter in range(dim_iteration):
                        for channel in range(channels_required):
                            head = channel * num_heads_per_bank + head_index_per_bank
                            if head > self.n_kv_heads - 1:
                                break
                            # for bank in range(self.num_banks):
                            #     dim = dim_iter * self.num_banks + bank
                            #     self.W_MEM_only_trace(channel, bank, row_current_head + dim_iter * rows_per_dim + row_offset, 1)
                            self.WR_ABK_only_trace(channel, row_current_head + dim_iter * rows_per_dim + row_offset, 1)
            else:
                # if banks_per_head < 16, channels_per_head < 1, one bank has more than one head, throw error
                # if 16 <= banks_per_head < 128, dim_iterations > 1, channels_per_head >= 1
                # if 128 <= banks_per_head < 512, dim_iterations = 1, devices_per_head = 1
                # if 512 <= banks_per_head, dim_iterations = 1, devices_per_head > 1
                                                                                                    # seqlen = 32k, head_dim = 128
                banks_per_head = (self.FC_total_banks - 1) // self.n_kv_heads + 1                   # 32, 256, 2k
                channels_per_head = (banks_per_head - 1) // (self.num_banks) + 1                    # 2,  16,  128
                devices_per_head = (channels_per_head - 1) // (self.num_channels) + 1               # 1,  1,   4
                # iteration along the head dimension
                dim_iterations = (self.head_dim - 1) // banks_per_head + 1                          # 4,  1,   1
                # iteration along the sequence dimension or rows per sequence
                rows_per_seq_iteration = (banks_per_head - 1) // self.head_dim + 1                  # 1,  2,   16
                seq_iterations = (seqlen - 1) // (self.DRAM_column * rows_per_seq_iteration) + 1    # 32, 16,  2
                rows_per_seq = (seqlen - 1) // (self.DRAM_column) + 1                               # 32, 32,  32
                channels_per_row_offset = (self.head_dim - 1) // self.num_banks + 1                 # 8
                for channel in range(channels_required):
                    if banks_per_head < self.num_banks:
                        # print("banks_per_head", banks_per_head)
                        raise ValueError("banks_per_head < self.num_banks. One head is mapped to less than one channel. Not enough channels are allocated.")
                    head = channel // (banks_per_head // self.num_banks)
                    if banks_per_head < 128:    # dim_iterations > 1, more than one dim in each row_offset are stored to a bank
                        for dim_iter in range(dim_iterations):   # E.g., head_dim = 128, banks_per_head = 32, channels_per_head = 2, dim_iterations = 128 / 32 = 4 in each bank. Within each iteration, Channel 0 is responsible for head 0: [0-15, 32-47, 64-79, 96-111], Channel 1 is responsible for head 1: [16-31, 48-63, 80-95, 112-127]. For bias vector, each head looks like (----CH0 16 Banks----,----CH1 16 Banks----) * 4.
                            row_offset = rows_per_seq - 1
                            self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset * dim_iterations + dim_iter, 1)
                    else:
                        # each head is mapped on a single device, channels_per_row_offset = 128 / 16 = 8
                        # E.g., head_dim = 128, banks_per_head = 256, channels_per_head = 16. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 7 is responsible for head 0: [0][112-127], Channel 8 is responsible for head 0: [1][0-15], Channel 9 is responsible for head 0: [1][16-31], ..., Channel 15 is responsible for head 0: [1][112-127]. For bias vector, each head has rows_per_seq_iteration = 2: (----CH0 16 Banks----) * 8, (----CH8 16 Banks----) * 8.
                        # each head is mapped on multiple devices
                        # E.g. head_dim = 128, banks_per_head = 2048, channels_per_head = 128. Channel 0 is responsible for head 0: [0][0-15], Channel 1 is responsible for head 0: [0][16-31], ..., Channel 127 is responsible for head 0: [15][112-127]. For bias vector, each head has rows_per_seq_iteration = 16: (----CH0 16 Banks----) * 128, ..., (----CH112 16 Banks----) * 128.
                        for bank in range(self.num_banks):
                            dim = ((channel % channels_per_head) % channels_per_row_offset) * self.num_banks + bank
                            if (channel % channels_per_head) // channels_per_row_offset == rows_per_seq - 1:
                                row_offset = rows_per_seq - 1
                                self.WR_ABK_only_trace(channel, self.cache_v_row_index + row_offset // rows_per_seq_iteration, 1)
            
            # Query x key_cache GEMV
            self.Vector_Matrix_Mul_score_pim_only_trace(self.cache_k_row_index, seqlen, "breakdown_sa_score")

        if self.trace_softmax:
            
        #     self.store_for_score_only_trace(self.scores_row_index, self.FC_total_banks, seqlen)
        #     self.SYNC_only_trace()
        #     self.load_for_score_only_trace(self.scores_row_index, self.FC_total_banks, seqlen)
        #     self.SYNC_only_trace()

        # if False:
            # CXL Port write scale
            rows_per_score = (seqlen - 1) // self.DRAM_column + 1
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * rows_per_score + seqlen // self.burst_length
            self.store_for_EWMUL_score_only_trace(channels_required, self.scores_row_index, total_banks, 0, seqlen)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * rows_per_score + seqlen // self.burst_length
            self.store_for_EWMUL_score_only_trace(channels_required, self.scores_row_index, total_banks, 1, seqlen)

            # Scale score
            num_scores_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
            for score_index in range(num_scores_per_bank):
                for row in range(rows_per_score):
                    if row == rows_per_score - 1:
                        offset = seqlen - row * self.DRAM_column
                    else:
                        offset = self.DRAM_column
                    self.EWMUL_only_trace(channel_lst, self.scores_row_index + score_index * rows_per_score + row, (offset - 1) // self.burst_length + 1)
            
            # CXL Port write mean of sum(exp)
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] * rows_per_score + seqlen // self.burst_length
            self.load_from_EWMUL_score_only_trace(channels_required, self.scores_row_index, total_banks, 2, seqlen)
            self.SYNC_only_trace()
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * rows_per_score + seqlen // self.burst_length
            self.store_for_EWMUL_score_only_trace(channels_required, self.scores_row_index, total_banks, 0, seqlen)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] * rows_per_score + seqlen // self.burst_length
            self.store_for_EWMUL_score_only_trace(channels_required, self.scores_row_index, total_banks, 1, seqlen)

            # Scale exp
            for score_index in range(num_scores_per_bank):
                for row in range(rows_per_score):
                    if row == rows_per_score - 1:
                        offset = seqlen - row * self.DRAM_column
                    else:
                        offset = self.DRAM_column
                    self.EWMUL_only_trace(channel_lst, self.scores_row_index + score_index * rows_per_score + row, (offset - 1) // self.burst_length + 1)

            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] * rows_per_score + seqlen // self.burst_length
            self.load_from_EWMUL_score_only_trace(channels_required, self.scores_row_index, total_banks, 2, seqlen)
            self.SYNC_only_trace()

        if self.trace_attention:
            # Score x value_cache GEMV
            self.Vector_Matrix_Mul_output_pim_only_trace(self.cache_v_row_index, seqlen, "breakdown_sa_output")

        # Output GEMV
        if self.trace_fc_kqvo:
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wo_row_index, self.dim, self.dim, FC_total_banks, "breakdown_sa_weight")
        if self.trace_norm:
            self.EWADD_only_trace(self.dim // self.burst_length)

            # RMSNorm   sa.pow   MAC_ABK
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
            self.store_for_neighbor_bank_input_only_trace(channels_required, input_vector_neighbor_bank_utilized_banks, 0, self.sa_copy_row_index, input_vector_neighbor_bank_length)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
            self.store_for_neighbor_bank_input_only_trace(channels_required, input_vector_neighbor_bank_utilized_banks, 1, self.sa_copy_row_index, input_vector_neighbor_bank_length)
            self.WR_BIAS_only_trace(channel_lst)
            self.MAC_ABK_only_trace(channel_lst, self.sa_copy_row_index, (input_vector_neighbor_bank_length - 1) // self.burst_length + 1, "breakdown_sa_pow")
            self.RD_MAC_only_trace(channel_lst)

            # CXL Port  
            # Reduction of dim // 16 intermidiate sum read from MAC
            # Broadcast a scalar to vector and store it for EWMUL
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.sa_copy_row_index, input_vector_EWMUL_length)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.sa_copy_row_index, input_vector_EWMUL_length)

            # RMSNorm   EWMUL
            self.EWMUL_only_trace(channel_lst, self.sa_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            for bank in range(self.num_banks):
                bank_group_index = 2
                if bank % 4 == bank_group_index:
                    self.COPY_BK_GB_only_trace(channel_lst, bank, self.sa_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
                    self.COPY_GB_BK_only_trace(channel_lst, bank-1, self.FFNNorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
            self.EWMUL_only_trace(channel_lst, self.FFNNorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            # Read RMSNorm result vector to GPR
            self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim // self.burst_length
            self.load_from_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.FFNNorm_row_index, input_vector_EWMUL_length)
            self.SYNC_only_trace()

        # w1 w3 FFN GEMV
        ffn_dim = self.w1.shape[0]
        ffn_bank_group_length = (ffn_dim - 1) // (total_banks // 4) + 1
        ffn_bank_group_utilized_banks = (ffn_dim - 1) // ffn_bank_group_length + 1
        if self.trace_fc_ffn:
            self.Vector_Matrix_Mul_weight_af_pim_only_trace(channel_lst, self.w1_row_index, self.dim, ffn_dim, FC_total_banks, "breakdown_ffn_weight")
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.w3_row_index, self.dim, ffn_dim, FC_total_banks, "breakdown_ffn_weight")

        # AF
        if self.trace_activation:
            iteration_required = ffn_dim > self.channels_per_block * (self.num_banks // 4) * self.DRAM_column
            if iteration_required:
                iteration_0 = total_banks // 4 * self.DRAM_column
                iteration_0_bank_group_length = (iteration_0 - 1) // (total_banks // 4) + 1
                iteration_0_bank_group_utilized_banks = (iteration_0 - 1) // iteration_0_bank_group_length + 1

                iteration_1 = ffn_dim - iteration_0
                iteration_1_bank_group_length = (iteration_1 - 1) // (total_banks // 4) + 1
                iteration_1_bank_group_utilized_banks = (iteration_1 - 1) // iteration_1_bank_group_length + 1

                self.store_for_EWMUL_input_only_trace(channels_required, iteration_0_bank_group_utilized_banks, 1, self.x1_row_index, iteration_0_bank_group_length)
                self.store_for_EWMUL_input_only_trace(channels_required, iteration_1_bank_group_utilized_banks, 1, self.x1_sigmoid_row_index, iteration_1_bank_group_length)
                self.store_for_EWMUL_input_only_trace(channels_required, iteration_0_bank_group_utilized_banks, 0, self.x1_row_index, iteration_0_bank_group_length)
                self.store_for_EWMUL_input_only_trace(channels_required, iteration_1_bank_group_utilized_banks, 0, self.x1_sigmoid_row_index, iteration_1_bank_group_length)
                self.EWMUL_only_trace(channel_lst, self.x1_row_index, (iteration_0_bank_group_length - 1) // self.burst_length + 1)
                self.EWMUL_only_trace(channel_lst, self.x1_sigmoid_row_index, (iteration_1_bank_group_length - 1) // self.burst_length + 1)

                for bank in range(self.num_banks):
                    bank_group_index = 2
                    if bank % 4 == bank_group_index:
                        self.COPY_BK_GB_only_trace(channel_lst, bank, self.x1_row_index, (iteration_0_bank_group_length - 1) // self.burst_length + 1)
                        self.COPY_BK_GB_only_trace(channel_lst, bank, self.x1_sigmoid_row_index, (iteration_1_bank_group_length - 1) // self.burst_length + 1)
                        self.COPY_GB_BK_only_trace(channel_lst, bank-1, self.x1_row_index, (iteration_0_bank_group_length - 1) // self.burst_length + 1)
                        self.COPY_GB_BK_only_trace(channel_lst, bank-1, self.x1_sigmoid_row_index, (iteration_1_bank_group_length - 1) // self.burst_length + 1)

                self.store_for_EWMUL_input_only_trace(channels_required, iteration_0_bank_group_utilized_banks, 1, self.x1_row_index, iteration_0_bank_group_length)
                self.store_for_EWMUL_input_only_trace(channels_required, iteration_1_bank_group_utilized_banks, 1, self.x1_sigmoid_row_index, iteration_1_bank_group_length)
                self.EWMUL_only_trace(channel_lst, self.x1_row_index, (iteration_0_bank_group_length - 1) // self.burst_length + 1)
                self.EWMUL_only_trace(channel_lst, self.x1_sigmoid_row_index, (iteration_1_bank_group_length - 1) // self.burst_length + 1)
                self.load_from_EWMUL_input_only_trace(channels_required, iteration_0_bank_group_utilized_banks, 2, self.x1_row_index, iteration_0_bank_group_length)
                self.load_from_EWMUL_input_only_trace(channels_required, iteration_1_bank_group_utilized_banks, 2, self.x1_sigmoid_row_index, iteration_1_bank_group_length)
            
            else:

                self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + ffn_bank_group_length * channels_required // self.burst_length
                self.store_for_EWMUL_input_only_trace(channels_required, ffn_bank_group_utilized_banks, 0, self.x1_sigmoid_row_index, ffn_bank_group_length)
                self.store_for_EWMUL_input_only_trace(channels_required, ffn_bank_group_utilized_banks, 1, self.x1_sigmoid_row_index, ffn_bank_group_length)
                self.EWMUL_only_trace(channel_lst, self.x1_sigmoid_row_index, (ffn_bank_group_length - 1) // self.burst_length + 1)

                for bank in range(self.num_banks):
                    bank_group_index = 2
                    if bank % 4 == bank_group_index:
                        self.COPY_BK_GB_only_trace(channel_lst, bank, self.x1_sigmoid_row_index, (ffn_bank_group_length - 1) // self.burst_length + 1)
                        self.COPY_GB_BK_only_trace(channel_lst, bank-1, self.x1_sigmoid_row_index, (ffn_bank_group_length - 1) // self.burst_length + 1)
                
                self.store_for_EWMUL_input_only_trace(channels_required, ffn_bank_group_utilized_banks, 0, self.x1_sigmoid_row_index, ffn_bank_group_length)
                self.EWMUL_only_trace(channel_lst, self.x1_sigmoid_row_index, (ffn_bank_group_length - 1) // self.burst_length + 1)
                self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + ffn_bank_group_length * channels_required // self.burst_length
                self.load_from_EWMUL_input_only_trace(channels_required, ffn_bank_group_utilized_banks, 2, self.x1_sigmoid_row_index, ffn_bank_group_length)
            self.SYNC_only_trace()

        # w2 FFN GEMV
        if self.trace_fc_ffn:
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.w2_row_index, ffn_dim, self.dim, FC_total_banks, "breakdown_ffn_weight")
        if self.trace_norm:
            self.EWADD_only_trace(self.dim // self.burst_length)

    def trace_only_embedding(self):
        bsz, _, _ = self.x.shape
        seqlen = self.seqlen
        total_banks = self.total_banks
        if self.model_parallel:
            FC_total_banks = total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = total_banks
            channels_required = self.channels_per_block
        channel_multi_transformer_block_required = self.num_channels // channels_required * channels_required
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]

        self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wq_row_index, self.vocab_size, self.dim, FC_total_banks, "breakdown_embedding_weight")
        # output embedding

        # RMSNorm   x.pow   MAC_ABK
        input_vector_MAB_BK_BK_length = (self.dim - 1) // (total_banks // 2) + 1
        self.WR_BIAS_only_trace(channel_lst)
        self.MAC_ABK_only_trace(channel_lst, self.x_row_index, (input_vector_MAB_BK_BK_length - 1) // self.burst_length + 1, "breakdown_sa_pow")
        self.RD_MAC_only_trace(channel_lst)

        # CXL Port  
        # Reduction of dim // 16 intermidiate sum read from MAC
        # Broadcast a scalar to vector and store it for EWMUL
        input_vector_EWMUL_length = (self.dim - 1) // (total_banks // 4) + 1
        input_vector_EWMUL_utilized_banks = (self.dim - 1) // input_vector_EWMUL_length + 1
        self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
        self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.x_copy_row_index, input_vector_EWMUL_length)

        # RMSNorm   EWMUL
        self.EWMUL_only_trace(channel_lst, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

        for bank in range(self.num_banks):
            bank_group_index = 2
            if bank % 4 == bank_group_index:
                self.COPY_BK_GB_only_trace(channel_lst, bank, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
                self.COPY_GB_BK_only_trace(channel_lst, bank-1, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
        self.EWMUL_only_trace(channel_lst, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

        # Read RMSNorm result vector to GPR
        self.time["RD_SBK"] += self.timing_constant["RD_SBK"] + self.dim // self.burst_length
        self.load_from_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 2, self.SANorm_row_index, input_vector_EWMUL_length)
        self.SYNC_only_trace()

        self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wo_row_index, self.dim, self.vocab_size, FC_total_banks, "breakdown_embedding_weight")

    def trace_only_FC(self):
        bsz, _, _ = self.x.shape
        seqlen = self.seqlen
        total_banks = self.total_banks
        if self.model_parallel:
            FC_total_banks = total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = total_banks
            channels_required = self.channels_per_block
        channel_multi_transformer_block_required = self.num_channels // channels_required * channels_required
        channel_lst = [channel for channel in range(channel_multi_transformer_block_required)]

        # RMSNorm

        # K/Q/V GEMV
        self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wq_row_index, self.dim, self.head_dim * self.n_heads, FC_total_banks, "breakdown_sa_weight")
        self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wk_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")
        self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wv_row_index, self.dim, self.head_dim * self.n_kv_heads, FC_total_banks, "breakdown_sa_weight")

        # Output GEMV
        self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wo_row_index, self.dim, self.dim, FC_total_banks, "breakdown_sa_weight")

        # w1 w3 FFN GEMV
        ffn_dim = self.w1.shape[0]
        self.Vector_Matrix_Mul_weight_af_pim_only_trace(channel_lst, self.w1_row_index, self.dim, ffn_dim, FC_total_banks, "breakdown_ffn_weight")
        self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.w3_row_index, self.dim, ffn_dim, FC_total_banks, "breakdown_ffn_weight")

        # w2 FFN GEMV
        self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.w2_row_index, ffn_dim, self.dim, FC_total_banks, "breakdown_ffn_weight")

    def memory_mapping(self):
        """
        Each chip has 1GB density (2 channels) w/ 32 banks (16 banks/channel).
        Each bank has 32MB and could store 16M BF16 values (2B), i.e., 16k rows and 1k columns.

        LLaMA-2-7B
        input: (1 x 4k) stored in 16 banks (256, 1 row) and broadcast to 2 channels
        wq wk wv: 32 heads (4k x 128, 1.5k row) stored in 2 channels, 32 banks
        xq xk xv: 32 heads (1 x 128, 3 row) stored in 2 channels, 32 banks
        cache_k: 32 heads (128 x L, 0.5k row) stored in 2 channels, 32 banks
        scores: 32 heads (1 x L, 4 row) stored in 2 channels, 32 banks
        cache_v: 32 heads (L x 128, 0.5k row) stored in 2 channels, 32 banks
        output: 32 heads (1 x 128, 1 row) stored in 2 channels, 32 banks
        wo: 32 heads (4k x 128, 0.5k row) stored in 2 channels, 32 banks
        sa: (1 x 4k) stored in 16 banks (256, 1 row) and broadcast to 2 channels
        w1 w3: 32 parts (4k x 344, 2*1376 rows) stored in 2 channels, 32 banks
        x1 x3: 32 parts (1 x 344, 2 rows) stored in 2 channels, 32 banks
        x1 sigmoid: 32 parts (1 x 344, 1 rows) stored in 2 channels, 32 banks
        x3: (1 x 11008) stored in 16 banks (688, 1 row) and broadcast to 2 channels
        w2: 32 parts (11008 x 128, 1376 rows) stored in 2 channels, 32 banks
        out: 32 parts (1 x 128, 1 rows) stored in 2 channels, 32 banks
        SiLU AF: TODO
        """
        self.dic_size = {}
        self.dic_row = {}
        self.dic_shape = {}
        self.dic_size["x"] = self.x.reshape(-1).shape[0]
        self.dic_row["x"] = (self.dic_size["x"] // self.num_banks - 1) // self.DRAM_column + 1
        total_banks = self.total_banks
        if self.model_parallel:
            FC_total_banks = total_banks * self.FC_devices
            channels_required = self.num_channels
        else:
            FC_total_banks = total_banks
            channels_required = self.channels_per_block
        assert self.dic_size["x"] == self.dim
        # print("x\t\t\t {} x {}\t\t\t requires {} rows".format(1, self.dic_size["x"], self.dic_row["x"]))
        # print("x_copy\t\t {} x {}\t\t\t requires {} rows".format(1, self.dic_size["x"], self.dic_row["x"]))
        # print("SANorm\t\t {} x {}\t\t\t requires {} rows".format(1, self.dic_size["x"], self.dic_row["x"]))
        # print("FFNNorm\t\t {} x {}\t\t\t requires {} rows".format(1, self.dic_size["x"], self.dic_row["x"]))

        self.dic_size["wq"] = self.wq.reshape(-1).shape[0]
        assert self.dic_size["wq"] == self.n_heads * self.head_dim * self.dim
        self.dic_row["wq"] = ((self.wq.shape[0] - 1) // FC_total_banks + 1) * ((self.wq.shape[1] - 1) // self.DRAM_column + 1)
        # print("wq\t\t\t {} x {} x {}\t requires {} rows".format(self.n_heads, self.dim, self.head_dim, self.dic_row["wq"]))
        self.dic_size["wk"] = self.wk.reshape(-1).shape[0]

        assert self.dic_size["wk"] == self.n_kv_heads * self.head_dim * self.dim
        self.dic_row["wk"] = ((self.wk.shape[0] - 1) // FC_total_banks + 1) * ((self.wk.shape[1] - 1) // self.DRAM_column + 1)
        # print("wk\t\t\t {} x {} x {} \t requires {} rows".format(self.n_heads, self.dim, self.head_dim, self.dic_row["wk"]))
        self.dic_size["wv"] = self.wv.reshape(-1).shape[0]
        assert self.dic_size["wv"] == self.n_kv_heads * self.head_dim * self.dim
        self.dic_row["wv"] = ((self.wv.shape[0] - 1) // FC_total_banks + 1) * ((self.wv.shape[1] - 1) // self.DRAM_column + 1)
        # print("wv\t\t\t {} x {} x {} \t requires {} rows".format(self.n_heads, self.dim, self.head_dim, self.dic_row["wv"]))

        self.dic_row["xq"] = 1
        self.dic_row["xk"] = 1
        self.dic_row["xv"] = 1

        self.dic_size["cache_k"] = self.max_seq_len * self.n_kv_heads * self.head_dim
        assert self.cache_k.reshape(-1).shape[0] == (self.start_pos + 1) * self.n_kv_heads * self.head_dim
        self.dic_row["cache_k"] = ((self.max_seq_len - 1) // self.FC_total_banks + 1) * ((self.n_kv_heads * self.head_dim - 1) // self.DRAM_column + 1)
        # print("cache_k\t\t {} x {} x {}\t\t requires {} rows".format(self.n_kv_heads, self.head_dim, "L", self.dic_row["cache_k"]))

        self.dic_size["scores"] = self.max_seq_len * self.n_kv_heads
        assert self.scores.reshape(-1).shape[0] == (self.start_pos + 1) * self.n_heads
        num_heads_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
        self.dic_row["scores"] = ((self.max_seq_len - 1) // self.DRAM_column + 1) * num_heads_per_bank
        # print("scores\t\t {} x {} x {}\t\t\t requires {} rows".format(self.n_heads, 1, "L", self.dic_row["scores"]))

        self.dic_size["cache_v"] = self.max_seq_len * self.n_kv_heads * self.head_dim
        assert self.cache_v.reshape(-1).shape[0] == (self.start_pos + 1) * self.n_kv_heads * self.head_dim
        if self.intra_device_attention:
            self.dic_row["cache_v"] = ((self.max_seq_len - 1) // self.DRAM_column + 1) * ((self.n_kv_heads - 1) // self.channels_per_block + 1) * ((self.head_dim - 1) // self.num_banks + 1)
        else:
            num_banks_per_head = (FC_total_banks - 1) // self.n_kv_heads + 1
            self.dic_row["cache_v"] = (self.max_seq_len - 1) // (self.DRAM_column * num_banks_per_head // self.head_dim) + 1
        # print("cache_v\t\t {} x {} x {}\t\t requires {} rows".format(self.n_kv_heads, "L", self.head_dim, self.dic_row["cache_v"]))

        self.dic_size["output"] = self.output.reshape(-1).shape[0]
        assert self.dic_size["output"] == self.n_heads * self.head_dim
        self.dic_row["output"] = (self.dic_size["output"] // total_banks - 1) // self.DRAM_column + 1
        # print("output\t\t {} x {}\t\t\t requires {} rows".format(1, self.n_heads * self.head_dim, self.dic_row["output"]))

        self.dic_size["wo"] = self.wo.reshape(-1).shape[0]
        assert self.dic_size["wo"] == self.n_heads * self.head_dim * self.dim
        self.dic_row["wo"] = ((self.wo.shape[0] - 1) // FC_total_banks + 1) * ((self.wo.shape[1] - 1) // self.DRAM_column + 1)
        # print("wo\t\t\t {} x {} x {}\t requires {} rows".format(self.n_heads, self.dim, self.head_dim, self.dic_row["wo"]))
        self.dic_size["sa"] = self.sa.reshape(-1).shape[0]
        assert self.dic_size["sa"] == self.n_heads * self.head_dim
        self.dic_row["sa"] = (self.dic_size["sa"] // total_banks - 1) // self.DRAM_column + 1
        # print("sa\t\t\t {} x {}\t\t\t requires {} rows".format(1, self.n_heads * self.head_dim, self.dic_row["sa"]))

        ffn_dim = self.w1.shape[0]
        ffn_parallel_dim = (ffn_dim - 1) // total_banks + 1
        ffn_FC_dim = (ffn_dim - 1) // FC_total_banks + 1

        self.dic_size["w1"] = self.w1.reshape(-1).shape[0]
        assert self.dic_size["w1"] == ffn_dim * self.dim
        self.dic_row["w1"] = ffn_FC_dim * ((self.dim - 1) // self.DRAM_column + 1)
        # print("w1\t\t\t {} x {} x {}\t requires {} rows".format(self.n_heads, self.dim, ffn_FC_dim, self.dic_row["w1"]))
        self.dic_size["w3"] = self.w3.reshape(-1).shape[0]
        assert self.dic_size["w3"] == ffn_dim * self.dim
        self.dic_row["w3"] = ffn_FC_dim * ((self.dim - 1) // self.DRAM_column + 1)
        # print("w3\t\t\t {} x {} x {}\t requires {} rows".format(self.n_heads, self.dim, ffn_FC_dim, self.dic_row["w3"]))
        self.dic_size["x1"] = ffn_dim
        self.dic_row["x1"] = (self.dic_size["x1"] // total_banks - 1) // self.DRAM_column + 1
        # print("x1\t\t\t {} x {} x {}\t\t requires {} rows".format(self.n_heads, 1, ffn_parallel_dim, self.dic_row["x1"]))
        self.dic_size["x3"] = ffn_dim
        self.dic_row["x3"] = (self.dic_size["x3"] // total_banks - 1) // self.DRAM_column + 1
        # print("x3\t\t\t {} x {} x {}\t\t requires {} rows".format(self.n_heads, 1, ffn_parallel_dim, self.dic_row["x3"]))
        self.dic_size["x1_sigmoid"] = ffn_dim
        self.dic_row["x1_sigmoid"] = (self.dic_size["x1_sigmoid"] // total_banks - 1) // self.DRAM_column + 1
        # print("x1_sigmoid\t {} x {} x {}\t\t requires {} rows".format(self.n_heads, 1, ffn_parallel_dim, self.dic_row["x1_sigmoid"]))
        self.dic_size["w2"] = self.w2.reshape(-1).shape[0]
        assert self.dic_size["w2"] == ffn_dim * self.dim
        self.dic_row["w2"] = ((self.dim - 1) // FC_total_banks + 1) * ((ffn_dim - 1) // self.DRAM_column + 1)
        # print("w2\t\t\t {} x {} x {}\t requires {} rows".format(self.n_heads, ffn_dim, self.head_dim, self.dic_row["w2"]))
        self.dic_size["ffn"] = self.ffn.reshape(-1).shape[0]
        assert self.dic_size["ffn"] == self.dim
        self.dic_row["ffn"] = (self.dic_size["ffn"] // total_banks - 1) // self.DRAM_column + 1
        # print("ffn\t\t\t {} x {}\t\t\t requires {} rows".format(1, self.dim, self.dic_row["ffn"]))

        size = sum([self.dic_size[key] for key in self.dic_size.keys()])
        rows = sum([self.dic_row[key] for key in self.dic_row.keys()])

        DIMMs_required = (channels_required - 1) // self.num_channels + 1
        # print("\nAllocated {} DIMMs {} Channels".format(DIMMs_required, channels_required))
        # print(size * 2 // (1024 * 1024), "MB are required in {} channels".format(self.channels_per_block))
        # print(rows, "rows are required in a bank")
        task_level_parallelism = (self.DRAM_row - rows) // (self.dic_row["cache_k"] + self.dic_row["cache_v"]) + 1
        # print(task_level_parallelism, "tasks are available to execute in parallel\n")
        # dimm_lst = ["dimm_" + str(i) for i in range(DIMMs_required)]
        # self.pim_device = {}
        # for dimm in dimm_lst:
        #     self.pim_device[dimm] = DIMM(args)


        # x in neighbor bank
        self.x_row_index = 0
        # x in a bank group
        self.x_copy_row_index = self.x_row_index + self.dic_row["x"]
        # SANorm
        self.SANorm_row_index = self.x_copy_row_index + self.dic_row["x"]
        # wq
        self.wq_row_index = self.SANorm_row_index + self.dic_row["x"]
        # wk
        self.wk_row_index = self.wq_row_index + self.dic_row["wq"]
        # wv
        self.wv_row_index = self.wk_row_index + self.dic_row["wk"]
        # xq
        self.xq_row_index = self.wv_row_index + self.dic_row["wv"]
        # xk
        self.xk_row_index = self.xq_row_index + self.dic_row["xk"]
        # cache_k
        self.cache_k_row_index = self.xk_row_index + self.dic_row["xq"]
        # scores
        self.scores_row_index = self.cache_k_row_index + self.dic_row["cache_k"]
        # cache_v
        self.cache_v_row_index = self.scores_row_index + self.dic_row["scores"]
        # output
        self.output_row_index = self.cache_v_row_index + self.dic_row["cache_v"]
        # wo
        self.wo_row_index = self.output_row_index + self.dic_row["output"]
        # sa
        self.sa_row_index = self.wo_row_index + self.dic_row["wo"]
        # sa copy
        self.sa_copy_row_index = self.sa_row_index + self.dic_row["sa"]
        # FFNNorm
        self.FFNNorm_row_index = self.sa_copy_row_index + self.dic_row["sa"]
        # w1
        self.w1_row_index = self.FFNNorm_row_index + self.dic_row["sa"]
        # w3
        self.w3_row_index = self.w1_row_index + self.dic_row["w1"]
        # x1
        self.x1_row_index = self.w3_row_index + self.dic_row["w3"]
        # x3
        self.x3_row_index = self.x1_row_index + self.dic_row["x1"]
        # x1_sigmoid
        self.x1_sigmoid_row_index = self.x3_row_index + self.dic_row["x3"]
        # ffn_vector
        self.dic_row["ffn_vector"] = (self.w1.shape[0] - 1) // (self.DRAM_column * self.num_banks) + 1
        self.ffn_vector_row_index = self.x1_sigmoid_row_index + self.dic_row["x1"]
        # w2
        self.w2_row_index = self.ffn_vector_row_index + self.dic_row["ffn_vector"]
        # ffn
        self.ffn_row_index = self.w2_row_index + self.dic_row["w2"]


    def memory_mapping_verification(self):
        flag = not self.only_trace and self.trace_prepare
        self.dic_shape["x_neighbor_bank"] = self.store_to_DRAM_multi_channel(self.x[0][0], self.x_row_index, "vector_neighbor_bank_0", flag)
        x_aim = self.load_from_DRAM_multi_channel(self.x.shape, self.x_row_index, "vector_neighbor_bank_0", self.dic_shape["x_neighbor_bank"][0], False)
        compare(self.x[0][0], x_aim[0][0], "x_neighbor_bank 0 mapping")
        
        # x in neighbor bank
        self.store_to_DRAM_multi_channel(self.x[0][0], self.x_row_index, "vector_neighbor_bank_1", flag)
        x_copy_aim = self.load_from_DRAM_multi_channel(self.x.shape, self.x_row_index, "vector_neighbor_bank_1", self.dic_shape["x_neighbor_bank"][0], False)
        compare(self.x[0][0], x_copy_aim[0][0], "x_neighbor_bank 1 mapping")

        # x in a bank group
        self.dic_shape["x_bank_group"] = self.store_to_DRAM_multi_channel(self.x[0][0], self.x_copy_row_index, "vector_bank_group_0", flag)
        x_copy_aim = self.load_from_DRAM_multi_channel(self.x.shape, self.x_copy_row_index, "vector_bank_group_0", self.dic_shape["x_bank_group"][0], False)
        compare(self.x[0][0], x_copy_aim[0][0], "x_bank_group mapping")

        # SANorm -> RMSNorm_x_aim
        self.dic_shape["SANorm_bank_group"] = self.store_to_DRAM_multi_channel(self.SANorm, self.SANorm_row_index, "vector_bank_group_0", False)
        SANorm_aim = self.load_from_DRAM_multi_channel(self.SANorm.shape, self.SANorm_row_index, "vector_bank_group_0", self.dic_shape["SANorm_bank_group"][0], False)
        compare(self.SANorm, SANorm_aim, "SANorm memory mapping")
        
        # wq
        self.dic_shape["wq"] = self.store_to_DRAM_multi_channel(self.wq, self.wq_row_index, self.mode["weights"], False)
        wq_aim = self.load_from_DRAM_multi_channel(self.wq.shape, self.wq_row_index, self.mode["weights"], self.dic_shape["wq"][0], False)
        compare(self.wq, wq_aim, "wq memory mapping")

        # wk
        self.dic_shape["wk"] = self.store_to_DRAM_multi_channel(self.wk, self.wk_row_index, self.mode["weights"], False)
        wk_aim = self.load_from_DRAM_multi_channel(self.wk.shape, self.wk_row_index, self.mode["weights"], self.dic_shape["wk"][0], False)
        compare(self.wk, wk_aim, "wk memory mapping")

        # wv
        self.dic_shape["wv"] = self.store_to_DRAM_multi_channel(self.wv, self.wv_row_index, self.mode["weights"], False)
        wv_aim = self.load_from_DRAM_multi_channel(self.wv.shape, self.wv_row_index, self.mode["weights"], self.dic_shape["wv"][0], False)
        compare(self.wv, wv_aim, "wv memory mapping")

        # cache_k
        bsz, seqlen, _, _ = self.cache_k.shape
        self.store_to_DRAM_multi_channel(self.cache_k[0], self.cache_k_row_index, self.mode["cache_k"], False)
        cache_k_aim = self.load_from_DRAM_multi_channel(self.cache_k.shape, self.cache_k_row_index, self.mode["cache_k"], seqlen, False)
        compare(cache_k_aim, self.cache_k, "cache_k memory mapping")

        # cache_v
        # print(self.cache_v.shape)
        cache_v = self.cache_v.transpose(1, 2).transpose(2, 3)
        # print(cache_v.shape)
        self.store_to_DRAM_multi_channel(cache_v[0], self.cache_v_row_index, self.mode["cache_v"], False)
        cache_v_aim = self.load_from_DRAM_multi_channel(cache_v.shape, self.cache_v_row_index, self.mode["cache_v"], seqlen, False).transpose(2, 3).transpose(1, 2)
        compare(cache_v_aim, self.cache_v, "cache_v memory mapping")

        # wo
        self.dic_shape["wo"] = self.store_to_DRAM_multi_channel(self.wo, self.wo_row_index, self.mode["weights"], False)
        wo_aim = self.load_from_DRAM_multi_channel(self.wo.shape, self.wo_row_index, self.mode["weights"], self.dic_shape["wo"][0], False)
        compare(self.wo, wo_aim, "wo memory mapping")

        # x in a bank group
        self.dic_shape["x_bank_group"] = self.store_to_DRAM_multi_channel(self.x[0][0], self.sa_row_index, "vector_bank_group_1", False)
        x_copy_aim = self.load_from_DRAM_multi_channel(self.x.shape, self.sa_row_index, "vector_bank_group_1", self.dic_shape["x_bank_group"][0], False)
        compare(self.x[0][0], x_copy_aim[0][0], "x mapping")

        # FFNNorm
        self.dic_shape["FFNNorm"] = self.store_to_DRAM_multi_channel(self.FFNNorm, self.FFNNorm_row_index, "vector_bank_group_0", False)
        FFNNorm_aim = self.load_from_DRAM_multi_channel(self.FFNNorm.shape, self.FFNNorm_row_index, "vector_bank_group_0", self.dic_shape["FFNNorm"][0], False)
        compare(self.FFNNorm, FFNNorm_aim, "FFNNorm memory mapping")

        # w1
        self.dic_shape["w1"] = self.store_to_DRAM_multi_channel(self.w1, self.w1_row_index, self.mode["weights"], False)
        w1_aim = self.load_from_DRAM_multi_channel(self.w1.shape, self.w1_row_index, self.mode["weights"], self.dic_shape["w1"][0], False)
        compare(self.w1, w1_aim, "w1 memory mapping")

        # w3
        self.dic_shape["w3"] = self.store_to_DRAM_multi_channel(self.w3, self.w3_row_index, self.mode["weights"], False)
        w3_aim = self.load_from_DRAM_multi_channel(self.w3.shape, self.w3_row_index, self.mode["weights"], self.dic_shape["w3"][0], False)
        compare(self.w3, w3_aim, "w3 memory mapping")
        
        # w2
        self.dic_shape["w2"] = self.store_to_DRAM_multi_channel(self.w2, self.w2_row_index, self.mode["weights"], False)
        w2_aim = self.load_from_DRAM_multi_channel(self.w2.shape, self.w2_row_index, self.mode["weights"], self.dic_shape["w2"][0], False)
        compare(self.w2, w2_aim, "w2 memory mapping")
