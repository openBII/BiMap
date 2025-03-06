import math
import torch
import torch.nn.functional as F
from aim_sim import PIM
from TransformerBlock import TransformerBlock

debug = True

class TransformerBlockGPT(TransformerBlock):
    """
    TransformerBlock Class inherits computate functionality from PIM class
    """
    def __init__(self, dic_model, args):
        super().__init__(dic_model, args)

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

        input_vector_neighbor_bank_length = (self.dim // self.TP_param - 1) // (self.total_banks // 2) + 1
        input_vector_neighbor_bank_utilized_banks = (self.dim // self.TP_param - 1) // input_vector_neighbor_bank_length + 1
        if self.trace_norm:
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 0, self.x_row_index, input_vector_neighbor_bank_length)
            self.store_for_neighbor_bank_input_only_trace(self.channels_per_block, input_vector_neighbor_bank_utilized_banks, 1, self.x_row_index, input_vector_neighbor_bank_length)

        # Norm   x.pow   MAC_ABK
        input_vector_MAB_BK_BK_length = (self.dim // self.TP_param - 1) // (total_banks // 2) + 1
        if self.trace_norm:
            self.WR_BIAS_only_trace(channel_lst)
            self.MAC_ABK_only_trace(channel_lst, self.x_row_index, (input_vector_MAB_BK_BK_length - 1) // self.burst_length + 1, "breakdown_sa_pow")
            self.RD_MAC_only_trace(channel_lst)

        # CXL Port  
        # Reduction of dim // 16 intermidiate sum read from MAC
        # Broadcast a scalar to vector and store it for EWMUL
        input_vector_EWMUL_length = (self.dim // self.TP_param - 1) // (total_banks // 4) + 1
        input_vector_EWMUL_utilized_banks = (self.dim // self.TP_param - 1) // input_vector_EWMUL_length + 1
        if self.trace_norm:
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
            # Standard Deviation   EWADD
            self.EWADD_only_trace(self.dim // self.TP_param)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.x_copy_row_index, input_vector_EWMUL_length)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.x_copy_row_index, input_vector_EWMUL_length)

            # Standard Deviation   EWMUL
            self.EWMUL_only_trace(channel_lst, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            for bank in range(self.num_banks):
                bank_group_index = 2
                if bank % 4 == bank_group_index:
                    self.COPY_BK_GB_only_trace(channel_lst, bank, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
                    self.COPY_GB_BK_only_trace(channel_lst, bank-1, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
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
            
            self.store_for_score_only_trace(self.scores_row_index, self.FC_total_banks, seqlen)
            self.SYNC_only_trace()
            self.load_for_score_only_trace(self.scores_row_index, self.FC_total_banks, seqlen)
            self.SYNC_only_trace()

        if False:
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
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wo_row_index, self.dim, self.head_dim * self.n_heads, FC_total_banks, "breakdown_sa_weight")
        if self.trace_norm:
            self.EWADD_only_trace(self.dim // self.burst_length)

            # RMSNorm   sa.pow   MAC_ABK
            input_vector_neighbor_bank_length = (self.dim // self.TP_param - 1) // (self.total_banks // 2) + 1
            input_vector_neighbor_bank_utilized_banks = (self.dim // self.TP_param - 1) // input_vector_neighbor_bank_length + 1
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
            self.store_for_neighbor_bank_input_only_trace(channels_required, input_vector_neighbor_bank_utilized_banks, 1, self.sa_copy_row_index, input_vector_neighbor_bank_length)
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
            self.store_for_neighbor_bank_input_only_trace(channels_required, input_vector_neighbor_bank_utilized_banks, 1, self.sa_copy_row_index, input_vector_neighbor_bank_length)
            self.WR_BIAS_only_trace(channel_lst)
            self.MAC_ABK_only_trace(channel_lst, self.sa_copy_row_index, (input_vector_neighbor_bank_length - 1) // self.burst_length + 1, "breakdown_sa_pow")
            self.RD_MAC_only_trace(channel_lst)

            # CXL Port  
            # Reduction of dim // 16 intermidiate sum read from MAC
            # Broadcast a scalar to vector and store it for EWMUL
            self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
            # Standard Deviation   EWADD
            self.EWADD_only_trace(self.dim // self.TP_param // self.burst_length)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.sa_copy_row_index, input_vector_EWMUL_length)
            self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.sa_copy_row_index, input_vector_EWMUL_length)

            # Standard Deviation   EWMUL
            self.EWMUL_only_trace(channel_lst, self.sa_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

            # division
            # Broadcast a scalar to vector and store it for EWMUL
            for bank in range(self.num_banks):
                bank_group_index = 2
                if bank % 4 == bank_group_index:
                    self.COPY_BK_GB_only_trace(channel_lst, bank, self.sa_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
                    self.COPY_GB_BK_only_trace(channel_lst, bank-1, self.sa_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
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

        # w1 FFN GEMV
        ffn_dim = self.w1.shape[0]
        ffn_bank_group_length = (ffn_dim - 1) // (total_banks // 4) + 1
        ffn_bank_group_utilized_banks = (ffn_dim - 1) // ffn_bank_group_length + 1
        if self.trace_fc_ffn:
            self.Vector_Matrix_Mul_weight_af_pim_only_trace(channel_lst, self.w1_row_index, self.dim, ffn_dim, FC_total_banks, "breakdown_ffn_weight")

            # w2 FFN GEMV
            self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.w2_row_index, ffn_dim * self.TP_param, self.dim // self.TP_param, FC_total_banks, "breakdown_ffn_weight")
        if self.trace_norm:
            self.EWADD_only_trace(self.dim // self.TP_param // self.burst_length)

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

        self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wq_row_index, self.vocab_size, self.dim // self.TP_param, FC_total_banks, "breakdown_embedding_weight")
        # output embedding

        # RMSNorm   x.pow   MAC_ABK
        input_vector_MAB_BK_BK_length = (self.dim // self.TP_param - 1) // (total_banks // 2) + 1
        self.WR_BIAS_only_trace(channel_lst)
        self.MAC_ABK_only_trace(channel_lst, self.x_row_index, (input_vector_MAB_BK_BK_length - 1) // self.burst_length + 1, "breakdown_sa_pow")
        self.RD_MAC_only_trace(channel_lst)

        # CXL Port  
        # Reduction of dim // 16 intermidiate sum read from MAC
        # Broadcast a scalar to vector and store it for EWMUL
        input_vector_EWMUL_length = (self.dim // self.TP_param - 1) // (total_banks // 4) + 1
        input_vector_EWMUL_utilized_banks = (self.dim // self.TP_param - 1) // input_vector_EWMUL_length + 1
        self.time["WR_SBK"] += self.timing_constant["WR_SBK"] + self.dim // self.burst_length
        # Standard Deviation   EWADD
        self.EWADD_only_trace(self.dim // self.TP_param)
        self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 0, self.x_copy_row_index, input_vector_EWMUL_length)
        self.store_for_EWMUL_input_only_trace(channels_required, input_vector_EWMUL_utilized_banks, 1, self.x_copy_row_index, input_vector_EWMUL_length)

        # Standard Deviation   EWMUL
        self.EWMUL_only_trace(channel_lst, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)

        # division
        # Broadcast a scalar to vector and store it for EWMUL
        for bank in range(self.num_banks):
            bank_group_index = 2
            if bank % 4 == bank_group_index:
                self.COPY_BK_GB_only_trace(channel_lst, bank, self.x_copy_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
                self.COPY_GB_BK_only_trace(channel_lst, bank-1, self.SANorm_row_index, (input_vector_EWMUL_length - 1) // self.burst_length + 1)
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

        self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wo_row_index, self.dim, self.vocab_size // self.TP_param, FC_total_banks, "breakdown_embedding_weight")


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
        self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.wo_row_index, self.dim, self.head_dim * self.n_heads, FC_total_banks, "breakdown_sa_weight")

        # w1 w3 FFN GEMV
        ffn_dim = self.w1.shape[0]
        self.Vector_Matrix_Mul_weight_af_pim_only_trace(channel_lst, self.w1_row_index, self.dim, ffn_dim, FC_total_banks, "breakdown_ffn_weight")

        # w2 FFN GEMV
        self.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, self.w2_row_index, ffn_dim * self.TP_param, self.dim // self.TP_param, FC_total_banks, "breakdown_ffn_weight")

    def memory_mapping(self):
        """
        Each chip has 1GB density (2 channels) w/ 32 banks (16 banks/channel).
        Each bank has 32MB and could store 16M BF16 values (2B), i.e., 16k rows and 1k columns.
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
        print("x\t\t\t {} x {}\t\t\t requires {} rows".format(1, self.dic_size["x"], self.dic_row["x"]))
        print("x_copy\t\t {} x {}\t\t\t requires {} rows".format(1, self.dic_size["x"], self.dic_row["x"]))
        print("SANorm\t\t {} x {}\t\t\t requires {} rows".format(1, self.dic_size["x"], self.dic_row["x"]))
        print("FFNNorm\t\t {} x {}\t\t\t requires {} rows".format(1, self.dic_size["x"], self.dic_row["x"]))

        self.dic_size["wq"] = self.wq.reshape(-1).shape[0]
        assert self.dic_size["wq"] == self.n_heads * self.head_dim * self.dim
        self.dic_row["wq"] = ((self.wq.shape[0] - 1) // FC_total_banks + 1) * ((self.wq.shape[1] - 1) // self.DRAM_column + 1)
        print("wq\t\t\t {} x {} x {}\t requires {} rows".format(self.n_heads, self.dim, self.head_dim, self.dic_row["wq"]))
        self.dic_size["wk"] = self.wk.reshape(-1).shape[0]

        assert self.dic_size["wk"] == self.n_kv_heads * self.head_dim * self.dim
        self.dic_row["wk"] = ((self.wk.shape[0] - 1) // FC_total_banks + 1) * ((self.wk.shape[1] - 1) // self.DRAM_column + 1)
        print("wk\t\t\t {} x {} x {} \t requires {} rows".format(self.n_heads, self.dim, self.head_dim, self.dic_row["wk"]))
        self.dic_size["wv"] = self.wv.reshape(-1).shape[0]
        assert self.dic_size["wv"] == self.n_kv_heads * self.head_dim * self.dim
        self.dic_row["wv"] = ((self.wv.shape[0] - 1) // FC_total_banks + 1) * ((self.wv.shape[1] - 1) // self.DRAM_column + 1)
        print("wv\t\t\t {} x {} x {} \t requires {} rows".format(self.n_heads, self.dim, self.head_dim, self.dic_row["wv"]))

        self.dic_row["xq"] = 1
        self.dic_row["xk"] = 1
        self.dic_row["xv"] = 1

        self.dic_size["cache_k"] = self.max_seq_len * self.n_kv_heads * self.head_dim
        assert self.cache_k.reshape(-1).shape[0] == (self.start_pos + 1) * self.n_kv_heads * self.head_dim
        self.dic_row["cache_k"] = ((self.max_seq_len - 1) // self.FC_total_banks + 1) * ((self.n_kv_heads * self.head_dim - 1) // self.DRAM_column + 1)
        print("cache_k\t\t {} x {} x {}\t\t requires {} rows".format(self.n_kv_heads, self.head_dim, "L", self.dic_row["cache_k"]))

        self.dic_size["scores"] = self.max_seq_len * self.n_kv_heads
        assert self.scores.reshape(-1).shape[0] == (self.start_pos + 1) * self.n_heads
        num_heads_per_bank = (self.n_heads - 1) // (self.channels_per_block * 4) + 1
        self.dic_row["scores"] = ((self.max_seq_len - 1) // self.DRAM_column + 1) * num_heads_per_bank
        print("scores\t\t {} x {} x {}\t\t\t requires {} rows".format(self.n_heads, 1, "L", self.dic_row["scores"]))
        
        self.dic_size["cache_v"] = self.max_seq_len * self.n_kv_heads * self.head_dim
        assert self.cache_v.reshape(-1).shape[0] == (self.start_pos + 1) * self.n_kv_heads * self.head_dim
        if self.intra_device_attention:
            self.dic_row["cache_v"] = ((self.max_seq_len - 1) // self.DRAM_column + 1) * ((self.n_kv_heads - 1) // self.channels_per_block + 1) * ((self.head_dim - 1) // self.num_banks + 1)
        else:
            num_banks_per_head = (FC_total_banks - 1) // self.n_kv_heads + 1
            self.dic_row["cache_v"] = (self.max_seq_len - 1) // (self.DRAM_column * num_banks_per_head // self.head_dim) + 1
        print("cache_v\t\t {} x {} x {}\t\t requires {} rows".format(self.n_kv_heads, "L", self.head_dim, self.dic_row["cache_v"]))

        self.dic_size["output"] = self.output.reshape(-1).shape[0]
        assert self.dic_size["output"] == self.TP_param * self.n_heads * self.head_dim
        self.dic_row["output"] = (self.dic_size["output"] // total_banks - 1) // self.DRAM_column + 1
        print("output\t\t {} x {}\t\t\t requires {} rows".format(1, self.n_heads * self.head_dim, self.dic_row["output"]))

        self.dic_size["wo"] = self.wo.reshape(-1).shape[0]
        assert self.dic_size["wo"] == self.n_heads * self.head_dim * self.dim
        self.dic_row["wo"] = ((self.wo.shape[0] - 1) // FC_total_banks + 1) * ((self.wo.shape[1] - 1) // self.DRAM_column + 1)
        print("wo\t\t\t {} x {} x {}\t requires {} rows".format(self.n_heads, self.dim, self.head_dim, self.dic_row["wo"]))
        self.dic_size["sa"] = self.sa.reshape(-1).shape[0]
        assert self.dic_size["sa"] == self.TP_param * self.n_heads * self.head_dim
        self.dic_row["sa"] = (self.dic_size["sa"] // total_banks - 1) // self.DRAM_column + 1
        print("sa\t\t\t {} x {}\t\t\t requires {} rows".format(1, self.n_heads * self.head_dim, self.dic_row["sa"]))

        ffn_dim = self.w1.shape[0]
        ffn_parallel_dim = (ffn_dim - 1) // total_banks + 1
        ffn_FC_dim = (ffn_dim - 1) // FC_total_banks + 1

        self.dic_size["w1"] = self.w1.reshape(-1).shape[0]
        assert self.dic_size["w1"] == ffn_dim * self.dim
        self.dic_row["w1"] = ffn_FC_dim * ((self.dim - 1) // self.DRAM_column + 1)
        print("w1\t\t\t {} x {} x {}\t requires {} rows".format(self.n_heads, self.dim, ffn_FC_dim, self.dic_row["w1"]))
        self.dic_size["x1"] = ffn_dim
        self.dic_row["x1"] = (self.dic_size["x1"] // total_banks - 1) // self.DRAM_column + 1
        print("x1\t\t\t {} x {} x {}\t\t requires {} rows".format(self.n_heads, 1, ffn_parallel_dim, self.dic_row["x1"]))
        self.dic_size["x1_sigmoid"] = ffn_dim
        self.dic_row["x1_sigmoid"] = (self.dic_size["x1_sigmoid"] // total_banks - 1) // self.DRAM_column + 1
        print("x1_sigmoid\t {} x {} x {}\t\t requires {} rows".format(self.n_heads, 1, ffn_parallel_dim, self.dic_row["x1_sigmoid"]))
        self.dic_size["w2"] = self.w2.reshape(-1).shape[0]
        assert self.dic_size["w2"] == ffn_dim * self.dim
        self.dic_row["w2"] = ((self.dim - 1) // FC_total_banks + 1) * ((ffn_dim - 1) // self.DRAM_column + 1)
        print("w2\t\t\t {} x {} x {}\t requires {} rows".format(self.n_heads, ffn_dim, self.head_dim, self.dic_row["w2"]))
        self.dic_size["ffn"] = self.ffn.reshape(-1).shape[0]
        assert self.dic_size["ffn"] == self.dim
        self.dic_row["ffn"] = (self.dic_size["ffn"] // total_banks - 1) // self.DRAM_column + 1
        print("ffn\t\t\t {} x {}\t\t\t requires {} rows".format(1, self.dim, self.dic_row["ffn"]))

        size = sum([self.dic_size[key] for key in self.dic_size.keys()])
        rows = sum([self.dic_row[key] for key in self.dic_row.keys()])

        DIMMs_required = (channels_required - 1) // self.num_channels + 1
        print("\nAllocated {} DIMMs {} Channels".format(DIMMs_required, channels_required))
        print(size * 2 // (1024 * 1024), "MB are required in {} channels".format(self.channels_per_block))
        print(rows, "rows are required in a bank")
        task_level_parallelism = (self.DRAM_row - rows) // (self.dic_row["cache_k"] + self.dic_row["cache_v"]) + 1
        print(task_level_parallelism, "tasks are available to execute in parallel\n")
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
        # x1
        self.x1_row_index = self.w1_row_index + self.dic_row["w1"]
        # x1_sigmoid
        self.x1_sigmoid_row_index = self.x1_row_index + self.dic_row["x1"]
        # ffn_vector
        self.dic_row["ffn_vector"] = (self.w1.shape[0] - 1) // (self.DRAM_column * self.num_banks) + 1
        self.ffn_vector_row_index = self.x1_sigmoid_row_index + self.dic_row["x1"]
        # w2
        self.w2_row_index = self.ffn_vector_row_index + self.dic_row["ffn_vector"]
        # ffn
        self.ffn_row_index = self.w2_row_index + self.dic_row["w2"]
