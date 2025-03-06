import os
import sys
import math
import pandas as pd
import argparse
import subprocess
import concurrent.futures
import matplotlib.pyplot as plt

# Add the directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'simulator_python')))

from cxl_latency import llama_latency, gpt_latency, vector_latency
from cent_power_calculator import DRAM_POWER, ACCEL_CYCLE, ACCEL_POWER, SRAM_POWER, CTRL_POWER, commands, isrs, power_calculator, command_processor

def get_args():
    parser = argparse.ArgumentParser('run_scripts.py')
    parser.add_argument("--num_channels", type=int, help="Number of channels per device", default=32)
    parser.add_argument("--num_devices", type=int, help="Number of CXL devices")
    parser.add_argument("--reuse_size", type=int, help="GB reuse size, depending on register number", default=32)
    parser.add_argument("--generate_trace_max_workers", type=int, help="maximum concurrent threads to generate traces, limited by memory", default=16)
    parser.add_argument("--seqlen", type=int, help="sequence length, using prompt + generation / 2", default=2048)
    parser.add_argument("--model", choices=["Llama2-7B", "Llama2-13B", "Llama2-70B"], help="LLM Model", required=True)
    parser.add_argument("--generate_trace", action="store_true", help="Generate traces")
    parser.add_argument("--simulate_trace", action="store_true", help="Simulate traces")
    parser.add_argument("--process_results", action="store_true", help="Process results")
    parser.add_argument("--update_csv", action="store_true", help="Update results to csv file")
    parser.add_argument(" ", action="store_true", help="Simulate traces")
    args = parser.parse_args()
    return args

args = get_args()


KILO = 1000
MEGA = 1000000
GIGA = 1000000000
FREQ = 2.00 * GIGA
WORD_SIZE = 256
tRC = 44.5
tBL = 1.25
tCCDL = 1.0

RV_COUNT = 8
# Latency of 1 SIMD operation
SB_RD_CYCLE = 1.00
SB_WR_CYCLE = 1.00
EXP_LANE_CYCLE = 11.00
RV_RMSNorm_CYCLE = 26.00
RV_ROTEmbed_CYCLE = 3.00 / RV_COUNT
RV_SFT_CYCLE_PIPELINE = 16.00 * SB_WR_CYCLE + 2.00 / RV_COUNT + 1.00 * SB_RD_CYCLE
RV_SFT_CYCLE_SINGLE = 16.00 * SB_WR_CYCLE + 2.00 + 1.00 * SB_RD_CYCLE

PCIE_ENERGY = 4.4

for accel_name in ["RED", "EXP", "VEC", "CTR"]:
    ACCEL_POWER[accel_name]["DYN"] = float(ACCEL_POWER[accel_name]["SWITCH"] + ACCEL_POWER[accel_name]["INT"])
    ACCEL_POWER[accel_name]["STT"] = float(ACCEL_POWER[accel_name]["LEAK"]) / float(GIGA)

InOut_latency = 0.15
n_heads = {"Llama2-7B": 32, "Llama2-13B": 40, "Llama2-70B": 64}
gqa_factor = {"Llama2-7B": 1, "Llama2-13B": 1, "Llama2-70B": 8}
embedding_size = {"Llama2-7B": 4096, "Llama2-13B": 5120, "Llama2-70B": 8192, "GPT3-175B": 12288, "GPT3-175B-TP-8": 12288, "OPT-66B": 9216}
ffn_size = {"Llama2-7B": 11008, "Llama2-13B": 13824, "Llama2-70B": 28672, "GPT3-175B": 12288*4, "GPT3-175B-TP-8": 12288*4, "OPT-66B": 9216*4}
filename = {"Llama2-7B": "../datasets/LLaMA-2-7B.pth", "Llama2-13B": "../datasets/LLaMA-2-13B.pth", "Llama2-70B": "../datasets/LLaMA-2-70B.pth"}
TransformerBlock_number = {"Llama2-7B": 32, "Llama2-13B": 40, "Llama2-70B": 80}
minimal_channel_per_block = {"Llama2-7B": 5, "Llama2-13B": 8, "Llama2-70B": 6}
if args.num_devices:
    device_list = [args.num_devices]
else:
    # device_list = [16, 32, 64, 128]
    # device_list = [i * 1 for i in range(1, 129)]
    # device_list = [i * 2 for i in range(1, 65)]
    device_list = [i * 4 for i in range(1, 33)]

# pipeline_parallelism = {"Llama2-70B": [80, 40, 20, 16, 10, 8, 5, 4, 2, 1]}
pipeline_parallelism = {"Llama2-70B": [80]}

if args.model == "GPT3-175B":
    model = "--GPT3-175B"
elif args.model == "Llama2-70B" or "Llama3" in args.model:
    model = "--Llama-GQA"
elif "Llama2" in args.model:
    model = "--Llama"
    

def calculate_acc_latency(args):
    latency = {}
    GQA_factor = 1.00 + 1.00 / gqa_factor[args.model]
    latency["RMSNorm_latency"] =  embedding_size[args.model] / 16.00 / 16.00 / args.num_channels * ACCEL_CYCLE["VEC"]    # EMB /16.00 /16.00 ADD
    latency["RMSNorm_latency"] += SB_RD_CYCLE + SB_WR_CYCLE + 1.00                              # 1 RED
    latency["RMSNorm_latency"] += RV_RMSNorm_CYCLE                                              # 1 RISCV
    latency["RMSNorm_latency"] = float(2.00 * latency["RMSNorm_latency"]) / float(FREQ / KILO)
    latency["Softmax_latency"] =  args.seqlen * n_heads[args.model] / 16.00 / args.num_channels * ACCEL_CYCLE["EXP"]        # TOK*HEAD /16.00 EXP
    latency["Softmax_latency"] += args.seqlen * n_heads[args.model] / 16.00 / args.num_channels * ACCEL_CYCLE["VEC"]        # TOK*HEAD /16.00 ADD
    latency["Softmax_latency"] += n_heads[args.model] * 1.00 * SB_RD_CYCLE                                     # HEAD RED
    latency["Softmax_latency"] += n_heads[args.model] * RV_SFT_CYCLE_PIPELINE                                  # HEAD RISCV
    latency["Softmax_latency"] = float(latency["Softmax_latency"]) / float(FREQ / KILO)
    latency["RotEmbed_latency"] = embedding_size[args.model] * RV_ROTEmbed_CYCLE                                 # EMB RISCV
    latency["RotEmbed_latency"] = float(GQA_factor * latency["RotEmbed_latency"]) / float(FREQ / KILO)
    return latency

def update_csv(args, device_list):

    '''
    Model: Llama2-7B, Llama2-13B, Llama2-70B
    Channels per device: 24, 32, 40, 48
    Channels per block: 32: {5, 7, 8, 10, 15, 16}
    '''
    # if os.path.exists('results.csv'):
    #     results_df = pd.read_csv('results.csv')
    # else:
    columns = ['Model', 'Device number', 'Data parallelism', 'Pipeline parallelism', 'Tensor parallelism', 'Channels per device', 'Channels per block', 'PIM latency', 'CXL latency', 'Acc latency', 'TransformerBlock latency', 'Embedding latency', 'Token latency', 'Throughput', 'Total power', 'Device utilization']
    results_df = pd.DataFrame(columns=columns)

    embedding_latency = {}
    embedding_compile_dir = f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel_embedding/{args.model}/"
    with open(f"{embedding_compile_dir}/compiled_results.txt", "r") as compiled_results_file:
        lines = compiled_results_file.readlines()
        for line in lines:
            filename, latency = line.split()[0], line.split()[1]
            channels_per_block = int(filename.split('_')[1])
            embedding_latency[channels_per_block] = float(latency)
    
    # FC_latency = {}
    # FC_compile_dir = f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel_FC/{args.model}/"
    # with open(f"{FC_compile_dir}/compiled_results.txt", "r") as compiled_results_file:
    #     lines = compiled_results_file.readlines()
    #     for line in lines:
    #         filename, latency = line.split()[0], line.split()[1]
    #         channels_per_block = int(filename.split('_')[1])
    #         FC_latency[channels_per_block] = float(latency)

    for num_device in device_list:
        PCIe_lanes = 144 // 32
        dp = 1
        for pp in pipeline_parallelism[args.model]:
            if pp < num_device // 2:
                continue
            elif pp > num_device // 2:
                tp = 1
                pp_per_device = (pp - 1) // num_device + 1
                blocks_per_device = pp_per_device * (TransformerBlock_number[args.model] // pp)
                channels_per_block = args.num_channels // blocks_per_device
                utilized_devices = (TransformerBlock_number[args.model] - 1) // blocks_per_device + 1
                device_utilization = 1.0 * utilized_devices / num_device
                if channels_per_block < minimal_channel_per_block[args.model]:
                    continue
                path = f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{args.seqlen}.txt.log"
                stats = command_processor(path)
                pim_latency = stats["latency"]
                cxl_latency = vector_latency(embedding_size[args.model], PCIe_lanes)
                acc_latency_dict = calculate_acc_latency(args)
                acc_latency = (acc_latency_dict["RMSNorm_latency"] + acc_latency_dict["Softmax_latency"] + acc_latency_dict["RotEmbed_latency"]) * blocks_per_device
                transformer_block_latency = pim_latency + cxl_latency + acc_latency
                token_latency = transformer_block_latency * TransformerBlock_number[args.model] + embedding_latency[channels_per_block] + InOut_latency
                throughput = 1000 / token_latency * pp

                energy_token = {}
                power_alldv = {}
                PCIE = embedding_size[args.model] if channels_per_block <= args.num_channels else embedding_size[args.model] * 10 + ffn_size[args.model] * 2
                energy_main, latency_main = power_calculator(stats, PCIE, n_heads[args.model], embedding_size[args.model], args.seqlen, gqa_factor[args.model])
                for comp in energy_main.keys():
                    energy_token[comp] = energy_main[comp] * utilized_devices
                    power_alldv[comp] = energy_main[comp] * utilized_devices / stats["latency"]
                total_energy = 0
                for comp in energy_token.keys():
                    total_energy += energy_token[comp]
                total_power = 0
                for comp in power_alldv.keys():
                    total_power += power_alldv[comp]
            else:
                devices_per_pp = num_device // pp
                tp = devices_per_pp
                utilized_devices = devices_per_pp * pp
                device_utilization = 1.0 * utilized_devices / num_device
                channels_per_block = args.num_channels
                path = f"../trace/{args.num_channels}_channels_per_device/model_parallel/{args.model}/trace_{devices_per_pp}_FC_devices.txt.log"
                stats = command_processor(path)
                pim_latency = stats["latency"]
                cxl_latency = llama_latency([embedding_size[args.model], ffn_size[args.model]], PCIe_lanes, devices_per_pp, devices_per_pp)
                acc_latency_dict = calculate_acc_latency(args)
                acc_latency = (acc_latency_dict["RMSNorm_latency"] + acc_latency_dict["Softmax_latency"] + acc_latency_dict["RotEmbed_latency"])
                transformer_block_latency = pim_latency + cxl_latency + acc_latency
                token_latency = transformer_block_latency * TransformerBlock_number[args.model] + embedding_latency[channels_per_block] + InOut_latency
                throughput = 1000 / token_latency * pp

                energy_token = {}
                power_alldv = {}
                PCIE = embedding_size[args.model] if channels_per_block <= args.num_channels else embedding_size[args.model] * 10 + ffn_size[args.model] * 2
                energy_main, latency_main = power_calculator(stats, PCIE, n_heads[args.model], embedding_size[args.model], args.seqlen, gqa_factor[args.model])
                for comp in energy_main.keys():
                    energy_token[comp] = energy_main[comp] * utilized_devices
                    power_alldv[comp] = energy_main[comp] * utilized_devices / stats["latency"]
                total_energy = 0
                for comp in energy_token.keys():
                    total_energy += energy_token[comp]
                total_power = 0
                for comp in power_alldv.keys():
                    total_power += power_alldv[comp]

            new_result = {
                'Model': args.model,
                'Device number': num_device,
                'Data parallelism': dp,
                'Pipeline parallelism': pp,
                'Tensor parallelism': tp,
                'Channels per device': args.num_channels,
                'Channels per block': channels_per_block,
                'PIM latency': pim_latency,
                'CXL latency': cxl_latency,
                'Acc latency': acc_latency,
                'TransformerBlock latency': transformer_block_latency,
                'Embedding latency': embedding_latency[channels_per_block],
                'Token latency': token_latency,
                'Throughput': throughput,
                'Total power': total_power,
                'Device utilization': device_utilization
            }
            # if device_utilization < 0.8:
            #     continue
            new_result_df = pd.DataFrame([new_result])

            # Check if the new result already exists in the DataFrame
            if not results_df[(results_df['Model'] == new_result['Model']) & 
                                (results_df['Device number'] == new_result['Device number']) &
                                (results_df['Data parallelism'] == new_result['Data parallelism']) &
                                (results_df['Pipeline parallelism'] == new_result['Pipeline parallelism']) &
                                (results_df['Tensor parallelism'] == new_result['Tensor parallelism']) &
                                (results_df['Channels per device'] == new_result['Channels per device']) & 
                                (results_df['Channels per block'] == new_result['Channels per block']) & 
                                (results_df['PIM latency'] == new_result['PIM latency']) &
                                (results_df['CXL latency'] == new_result['CXL latency']) & 
                                (results_df['Acc latency'] == new_result['Acc latency']) &
                                (results_df['TransformerBlock latency'] == new_result['TransformerBlock latency']) &
                                (results_df['Embedding latency'] == new_result['Embedding latency']) &
                                (results_df['Token latency'] == new_result['Token latency']) &
                                (results_df['Throughput'] == new_result['Throughput']) &
                                (results_df['Total power'] == new_result['Total power']) &
                                (results_df['Device utilization'] == new_result['Device utilization'])
                                ].empty:                                   
                print("Duplicate result, not appending.")
            else:
                results_df = pd.concat([results_df, new_result_df], ignore_index=True)

        tmp_num_device = num_device * 1
        if tmp_num_device > 16:
            tmp_num_device //= 2
            dp *= 2
            # print(f"device number: {num_device}, tmp_num_device: {tmp_num_device}, dp: {dp}")
            result_df_tmp = results_df[
                (results_df['Model'] == args.model) & 
                (results_df['Device number'] == tmp_num_device) &
                (results_df['Channels per device'] == args.num_channels)
            ]
            index = result_df_tmp.index
            for i in range(len(index)):
                new_result_df = result_df_tmp.iloc[[i]].copy()  # Copy the i-th row to new_result_df
                new_result_df.loc[index[i], 'Device number'] = num_device
                new_result_df.loc[index[i], 'Data parallelism'] = result_df_tmp.loc[index[i], 'Data parallelism'] * dp
                new_result_df.loc[index[i], 'Throughput'] = result_df_tmp.loc[index[i], 'Throughput'] * dp
                new_result_df.loc[index[i], 'Device utilization'] = result_df_tmp.loc[index[i], 'Device utilization'] * dp * (num_device // dp) / num_device

                if not results_df[(results_df['Model'] == new_result_df['Model'][index[i]]) & 
                                    (results_df['Device number'] == new_result_df['Device number'][index[i]]) &
                                    (results_df['Data parallelism'] == new_result_df['Data parallelism'][index[i]]) &
                                    (results_df['Pipeline parallelism'] == new_result_df['Pipeline parallelism'][index[i]]) &
                                    (results_df['Tensor parallelism'] == new_result_df['Tensor parallelism'][index[i]]) &
                                    (results_df['Channels per device'] == new_result_df['Channels per device'][index[i]]) & 
                                    (results_df['Channels per block'] == new_result_df['Channels per block'][index[i]]) & 
                                    (results_df['PIM latency'] == new_result_df['PIM latency'][index[i]]) &
                                    (results_df['CXL latency'] == new_result_df['CXL latency'][index[i]]) & 
                                    (results_df['Acc latency'] == new_result_df['Acc latency'][index[i]]) &
                                    (results_df['TransformerBlock latency'] == new_result_df['TransformerBlock latency'][index[i]]) &
                                    (results_df['Embedding latency'] == new_result_df['Embedding latency'][index[i]]) &
                                    (results_df['Token latency'] == new_result_df['Token latency'][index[i]]) &
                                    (results_df['Throughput'] == new_result_df['Throughput'][index[i]]) &
                                    (results_df['Total power'] == new_result_df['Total power'][index[i]]) &
                                    (results_df['Device utilization'] == new_result_df['Device utilization'][index[i]])
                                    ].empty:                           
                    print("Duplicate result, not appending.")
                else:
                    results_df = pd.concat([results_df, new_result_df], ignore_index=True)

    # Save the DataFrame to a CSV file
    results_df.to_csv('results_various_DP.csv', index=False)
    # print(results_df)

    # results_df = results_df.drop_duplicates(subset='Throughput')

update_csv(args, device_list)
    

