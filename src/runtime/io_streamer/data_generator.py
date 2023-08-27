# Copyright (C) OpenBII
# Team: CBICR
# SPDX-License-Identifier: Apache-2.0
# See: https://spdx.org/licenses/

from time import sleep
from PIL import Image
import numpy as np
import argparse
import os

import src.compiler.ir.msg_pb2 as pb
import src.compiler.ir.basic_pb2 as tianjic_basic
import toml

def divide_into_blocks(img, ioconfig):
    for ioblock in ioconfig:
        bnx = ioblock.begin_position.nx
        bny = ioblock.begin_position.ny
        bnf = ioblock.begin_position.nf
        enx = ioblock.end_position.nx+1 #[b, e]->[b, e+1)
        eny = ioblock.end_position.ny+1
        enf = ioblock.end_position.nf+1
        block = img[bnx:enx, bny:eny, bnf:enf]
        block.tofile(str(ioblock.id))
    return 

def get_image(path = "style.png"):
    img = Image.open(path)
    img = np.array(img, np.int8)
    return img

def get_config(path = None):
    ioblock = None
    ioblock.begin_position.nx = 0
    ioblock.begin_position.ny = 0
    ioblock.begin_position.nf = 0
    ioblock.end_position.nx = 5
    ioblock.end_position.ny = 5
    ioblock.end_position.nf = 1
    ioblock.id = 123
    ioconfig = [ioblock]

    return ioconfig

def main(img_path, config_path):
    img = get_image(img_path)
    ioconfig = get_config(config_path )
    divide_into_blocks(img, ioconfig)

def fetch(req):
    data_root = toml.load("top/config.toml")["path"]["data"]
    data_path = data_root + req.case_name + '/' + req.id.split('.')[0] + '.dat'
    
    if not os.path.exists(data_path):
        # data_gen(req.memory_block_id, req.pic, req.shape, req.seed)
        print ("{} not exists".format(data_path))
        return np.array([], np.int8).tobytes()


    #raw_data， int32
    raw = np.fromfile(data_path, dtype=np.int32)
    data = []
    if req.precision == tianjic_basic.Precision.INT_32:
        data = raw
    elif req.precision == tianjic_basic.Precision.TERNARY:
        i = 0
        while i < len(raw):
            n = 0
            for j in range(4):
                n <<= 2
                jj = 3 - j
                if raw[i+jj] == -1:
                    raw[i+jj] = 3
                n |= (raw[i+jj] & 0x3)   
            i += 4
            data.append(n)
        data = np.array(data, np.int8)
    else:
        data = raw.astype(np.int8)
    
    return data.tobytes()

# TODO : Test data_gen
def data_gen(bid, pic, shape, seed):
    np.random.seed(seed)
    if pic == 0x2:
        data_gen_02(bid, shape)

def data_gen_02(bid, ioblock):
    bnx = ioblock.begin_position.nix
    bny = ioblock.begin_position.niy
    bnf = ioblock.begin_position.nf
    enx = ioblock.end_position.nix+1 #[b, e]->[b, e+1)
    eny = ioblock.end_position.niy+1
    enf = ioblock.end_position.nf+1

    nx = enx - bnx - 1
    ny = eny - bny - 1
    nf = enf - bnf - 1

    nbit = None
    dtype = None
    if ioblock.precision in [tianjic_basic.Precision.UINT_8, tianjic_basic.Precision.INT_8]:
        nbit = 8
        dtype = 'uint8' 
    else :
        nbit = 32
        dtype = 'int32'

    # 32B  = 256 bit
    bits = nf*nbit
    bits  =  np.ceil(bits / 256) * 256
    nf_ = int(bits // nbit)

    data = np.random.randint(0, 1<<nbit, (ny, nx, nf))

    zeros = np.zeros((ny, nx, nf_), dtype=dtype)
    zeros[:,:,0:nf] = data
    zeros.tofile(bid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--config_path', type=str, default=None)
    args = parser.parse_args()	
    main(args.img_path, args.config_path)


