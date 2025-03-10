import os
import math
import pandas as pd
import argparse
import subprocess
import concurrent.futures
from cent_simulation.cxl_latency import llama_latency, gpt_latency, vector_latency
from cent_simulation.cent_power_calculator import DRAM_POWER, ACCEL_CYCLE, ACCEL_POWER, SRAM_POWER, CTRL_POWER, commands, isrs, power_calculator, command_processor, KILO, MEGA, GIGA, FREQ, WORD_SIZE, tRC, tBL, tCCDL, RV_COUNT, SB_RD_CYCLE, SB_WR_CYCLE, EXP_LANE_CYCLE, RV_RMSNorm_CYCLE, RV_ROTEmbed_CYCLE, RV_SFT_CYCLE_PIPELINE, RV_SFT_CYCLE_SINGLE
from cent_simulation.utils import InOut_latency, n_heads, gqa_factor, embedding_size, ffn_size, TransformerBlock_number, minimal_channel_per_block, pipeline_parallel_mode_list, model_parallel_mode_list
from cent_simulation.run_sim import factorize
from cent_simulation.function_sim import func_sim_llama

def generate_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_channels", type=int, help="Number of channels per device", default=32)
    parser.add_argument("--num_devices", type=int, help="Number of CXL devices", default=32)
    parser.add_argument("--PCIE_lanes", type=int, help="Number of PCIE lanes", default=144)
    parser.add_argument("--reuse_size", type=int, help="GB reuse size, depending on register number", default=32)
    parser.add_argument("--generate_trace_max_workers", type=int, help="maximum concurrent threads to generate traces, limited by memory", default=8)
    parser.add_argument("--run_simulation_max_workers", type=int, help="maximum concurrent threads to generate traces, limited by memory", default=4)
    parser.add_argument("--model", choices=["Llama2-7B", "Llama2-13B", "Llama2-70B"], help="LLM Model", default="Llama2-7B")
    parser.add_argument("--generate_trace", action="store_true", help="Generate traces")
    parser.add_argument("--simulate_trace", action="store_true", help="Simulate traces")
    parser.add_argument("--process_results", action="store_true", help="Process results")
    parser.add_argument("--update_csv", action="store_true", help="Update results to csv file")
    parser.add_argument("--simulation_result_path", type=str, help="Path to the result file", default="simulation_results.csv")
    parser.add_argument("--process_throughputs", action="store_true", help="average throughputs for various seqlen")
    parser.add_argument("--processed_result_path", type=str, help="Path to the final result file", default="processed_results.csv")
    parser.add_argument("--phase", choices=["end2end", "prefill", "decoding"], help="Phase of the model", default="end2end")
    parser.add_argument("--prefill", type=int, help="Prefill length", default=1024)
    parser.add_argument("--decoding", type=int, help="Decoding length", default=8192)
    parser.add_argument("--seqlen", type=int, nargs='+', help="Sequence list")
    parser.add_argument("--seqlen_gap", type=int, help="Gap between sequence lengths", default=1024)
    args = parser.parse_args()
    return args

def generate_trace(args, seqlen_list):
    
    model_parallel = False
    rel_path = "ext/cent_pim/cent_simulation/"

    print(f"Generating traces for {args.model} with {args.generate_trace_max_workers} threads...")

    if args.model == "GPT3-175B":
        model = "--GPT3-175B"
    elif args.model == "Llama2-70B" or "Llama3" in args.model:
        model = "--Llama-GQA"
    elif "Llama2" in args.model:
        model = "--Llama"

    commands_generate_traces = []
    blocks_per_device = (TransformerBlock_number[args.model] - 1) // args.num_devices + 1
    channels_per_block = args.num_channels // blocks_per_device
    FC_devices_list = factorize(args.num_devices)
    trace_log_file = []
    for seqlen in seqlen_list:
        if model_parallel:
            filename = f"./trace/{args.num_channels}_channels_per_device/model_parallel/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"
        else:
            filename = f"./trace/{args.num_channels}_channels_per_device/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"
        trace_log_file.append(filename)
    
    # No Embedding
    for seqlen in seqlen_list:
        if model_parallel:
            for FC_devices in FC_devices_list:
                if not os.path.exists(f"./trace/{args.num_channels}_channels_per_device/model_parallel/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"):
                    commands_generate_traces.append(["python", rel_path + "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--only-trace", "--num-channels", str(args.num_channels), "--FC-devices", str(FC_devices), "--model-parallel", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"../trace/{args.num_channels}_channels_per_device/model_parallel/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"])
                if not os.path.exists(f"./trace/{args.num_channels}_channels_per_device/model_parallel_FC/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"):
                    commands_generate_traces.append(["python", rel_path + "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--only-FC", "--only-trace", "--num-channels", str(args.num_channels), "--FC-devices", str(FC_devices), "--model-parallel", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"../trace/{args.num_channels}_channels_per_device/model_parallel_FC/{args.model}/trace_{FC_devices}_FC_devices_seqlen_{seqlen}.txt"])
        else:
            if channels_per_block < minimal_channel_per_block[args.model]:
                raise ValueError(f"Channels per block {channels_per_block} is less than minimal channel per block {minimal_channel_per_block[args.model]}")
            if not os.path.exists(f"./trace/{args.num_channels}_channels_per_device/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"):
                commands_generate_traces.append(["python", rel_path + "function_sim.py", model, "--n_heads", str(n_heads[args.model]), "--ffn_dim", str(ffn_size[args.model]), "--only-trace", "--num-channels", str(args.num_channels), "--channels-per-block", str(channels_per_block), "--pipeline-parallel", "--multi-tb-per-device", "--seqlen", str(seqlen), "--op-trace", "--GEMV", "reuse-GB", "--reuse-size", str(args.reuse_size), "--trace-file", f"../trace/{args.num_channels}_channels_per_device/pipeline_parallel/{args.model}/trace_{channels_per_block}_channels_per_block_seqlen_{seqlen}.txt"])
    
    func_sim_llama(trace_log_file, seqlen_list)

def DRAM_PIM_Compute_API():
    
    # Prepare arguments & directories
    args = generate_args()
    if args.seqlen:
        seqlen_list = args.seqlen
    else:
        seqlen_list = [i * args.seqlen_gap for i in range(1, (args.prefill + args.decoding) // args.seqlen_gap + 1)]
    for mode in pipeline_parallel_mode_list + model_parallel_mode_list:
        subprocess.run(["mkdir", "-p", f"./trace/{args.num_channels}_channels_per_device/{mode}/{args.model}"])
    
    # Generate traces
    generate_trace(args, seqlen_list)
        
if __name__ == "__main__":
    print("Hello CENT!")
    DRAM_PIM_Compute_API()