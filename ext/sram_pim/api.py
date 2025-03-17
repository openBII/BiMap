import math

# [ISSCC'23] 7.2 A 28nm 64-kb 31.6-TFLOPS/W Digital-Domain Floating-Point- 
# Computing-Unit and Double-Bit 6T-SRAM Computing-in- 
# Memory Macro for Floating-Point CNNs (BF16)

max_width = 2048 * 8 / 16 # bits per bank

spec = {
    "size": 64 * 1024, # bit = 64 * 64 * 16b
    "shape": [128, 8, 4], # input, output, parallel (Limited by CAT) = 512 -> 8
    "channel": 128, # limited by CAT
    "area": 0.146, # mm2
    "mac_access_latency": 6.8, # ns
    "power_eff": [14.04, 31.6], # (0.9V and 0.6V) TFLOPS/W
    "area_eff": 2.05 # TFLOPS/mm2
}

def SRAM_PIM_Compute_API(input_size, output_size, macro_num, batch_num=1, split=['o']):

    max_power = spec["area"] * spec["area_eff"] / spec["power_eff"][0]
    min_power = spec["area"] * spec["area_eff"] / spec["power_eff"][1]
    print("\nTask: ", input_size, "->", output_size)
    print("Area: ", spec["area"] * macro_num, "mm2")
    print("Power(Min): ", min_power * macro_num, "W")
    print("Power(Max): ", max_power * macro_num, "W")
    
    input_width = macro_num * spec["shape"][0] * spec["shape"][2]
    reduce_size = math.ceil(input_size / input_width)
    reload_size = math.ceil(output_size / spec["shape"][1])
    print("Input Split (Reduce): ", reduce_size, str(input_width)+"x")
    print("Output Split (Reload): ", reload_size)
    
    if input_width * 16 > max_width:
        load_inp_round = math.ceil(input_width * 16 / max_width)
        print("Load Input Round: ", load_inp_round)
    else:
        load_inp_round = 1
        print("Load Input Round: ", load_inp_round)
    
    if spec["shape"][1] * spec["shape"][0] * spec["shape"][2] * 16 > max_width:
        load_wgt_round = math.ceil(spec["shape"][1] * spec["shape"][0] * spec["shape"][2] * 16 / max_width)
        print("Load Weight Round: ", load_wgt_round)
    else:
        load_wgt_round = 1
        print("Load Weight Round: ", load_wgt_round)
    
    # 2048 bit = 16bit * 128
    ld_i_latency = spec["mac_access_latency"]
    ld_w_latency = spec["mac_access_latency"] * spec["shape"][1]
    print("Load Latency / Round: ", spec["mac_access_latency"], "ns")
    print("Load Weight / Round: ", spec["mac_access_latency"] * spec["shape"][1], "ns")
    
    exe_latency = spec["size"] / 16 / (spec["area_eff"] * spec["area"] * 1e12) * 1e9
    print("MAC Execute Latency / Round: ", exe_latency, "ns")
    
    overall_latency = reload_size * reduce_size * (ld_i_latency * load_inp_round * batch_num + ld_w_latency * load_wgt_round + exe_latency)
    overall_no_pipe_latency = reload_size * reduce_size * (ld_i_latency * load_inp_round * batch_num + ld_w_latency * load_wgt_round + exe_latency * load_inp_round * batch_num)
    
    print("Overall Latency (Pipeline): ", overall_latency, "ns")
    print("Overall Latency (No Pipe) : ", overall_no_pipe_latency, "ns")
    print("Latency / Batch (Pipeline): ", overall_latency / batch_num, "ns")
    print("Latency / Batch (No Pipe) : ", overall_no_pipe_latency / batch_num, "ns")
    

if __name__ == "__main__":
    batch_num = 1
    SRAM_PIM_Compute_API(4096, 8, 8, batch_num)
    SRAM_PIM_Compute_API(4096, 22, 8, batch_num)
    SRAM_PIM_Compute_API(4096, 22, 8, batch_num)
    SRAM_PIM_Compute_API(11008, 8, 8, batch_num)