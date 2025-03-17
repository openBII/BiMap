#!/usr/bin/python
# -*-coding:utf-8-*-
import torch
import sys
import os
import math
import argparse
import numpy as np
import torch
import collections
import configparser
import time
from importlib import import_module
from ext.mnsim2.MNSIM.Interface.interface import *
from ext.mnsim2.MNSIM.Accuracy_Model.Weight_update import weight_update
from ext.mnsim2.MNSIM.Mapping_Model.Behavior_mapping import behavior_mapping
from ext.mnsim2.MNSIM.Mapping_Model.Tile_connection_graph import TCG
from ext.mnsim2.MNSIM.Latency_Model.Model_latency import Model_latency
from ext.mnsim2.MNSIM.Area_Model.Model_Area import Model_area
from ext.mnsim2.MNSIM.Power_Model.Model_inference_power import Model_inference_power
from ext.mnsim2.MNSIM.Energy_Model.Model_energy import Model_energy

def generate_structure_file(nf: int, nr: int, xbar_size: int = 256):
                
    layer_info = collections.OrderedDict()
    
    # Shape Info (GeMV -> Conv)
    layer_info['type'] = 'conv'
    layer_info['Inputchannel'] = nr
    layer_info['Inputsize'] = [1, 1]
    layer_info['Kernelsize'] = 1
    layer_info['Stride'] = 1
    layer_info['Padding'] = 0
    layer_info['Depthwise'] = 'normal'
    layer_info['Outputchannel'] = nf
    layer_info['Outputsize'] = [1, 1]
    
    layer_info['Inputbit'] = 9
    layer_info['Weightbit'] = 9
    layer_info['outputbit'] = 9
    
    channel_N = xbar_size
    complete_bar_num = layer_info['Inputchannel'] // channel_N
    residual_col_num = layer_info['Inputchannel'] % channel_N
            
    in_channels_list = []
    if residual_col_num > 0:
        in_channels_list = [channel_N] * complete_bar_num + [residual_col_num]
    else:
        in_channels_list = [channel_N] * complete_bar_num
    layer_info['row_split_num'] = len(in_channels_list)
    
    device_bit = 2 # Default
    hw_weight_bit = math.floor(math.log2(device_bit))
    layer_info['weight_bit_split_part'] = math.ceil((layer_info['Weightbit'] - 1) / hw_weight_bit)
    
    layer_info['Inputindex'] = [-1]
    layer_info['Outputindex'] = []
    layer_info['Layerindex'] = 0
    
    # Merge Structure Info
    net_tupe = (layer_info, [])
    net_array = [[net_tupe]]
    return net_array

def PIM_Compute_API(nf: int, nr: int, xbar_size: int = 64):
    
    # Load Config
    SimConfig_path = os.path.join("ext/mnsim2/SimConfig.ini")
    structure_file = generate_structure_file(nf, nr, xbar_size)
    TCG_mapping = TCG(structure_file, SimConfig_path)
    
    # Latency
    __latency = Model_latency(NetStruct=structure_file, SimConfig_path=SimConfig_path, TCG_mapping=TCG_mapping)
    __latency.calculate_model_latency(mode=1)
    
    # Power
    __power = Model_inference_power(NetStruct=structure_file, SimConfig_path=SimConfig_path, TCG_mapping=TCG_mapping)
    __power.calculate_model_power()
    
    # Area
    __area = Model_area(NetStruct=structure_file, SimConfig_path=SimConfig_path, TCG_mapping=TCG_mapping)
    __area.model_area_output(1,1)
    
    # Final Latency
    final_latency = max(max(__latency.finish_time))
    total_power = __power.arch_total_power
    return final_latency, total_power

# Main for Testing
if __name__ == "__main__":
    final_latency, total_power = PIM_Compute_API(512, 64)
    print("Final Latency: {} (ns)".format(final_latency))
    print("Total Power: {}".format(total_power))