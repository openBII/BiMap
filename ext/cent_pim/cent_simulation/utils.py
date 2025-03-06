import torch
import argparse

# debug = True
debug = False

InOut_latency = 0.15        # Top-K sampling on CPU
n_heads = {"Llama2-7B": 32, "Llama2-13B": 40, "Llama2-70B": 64}
gqa_factor = {"Llama2-7B": 1, "Llama2-13B": 1, "Llama2-70B": 8}
embedding_size = {"Llama2-7B": 4096, "Llama2-13B": 5120, "Llama2-70B": 8192, "GPT3-175B": 12288, "GPT3-175B-TP-8": 12288, "OPT-66B": 9216}
ffn_size = {"Llama2-7B": 11008, "Llama2-13B": 13824, "Llama2-70B": 28672, "GPT3-175B": 12288*4, "GPT3-175B-TP-8": 12288*4, "OPT-66B": 9216*4}
TransformerBlock_number = {"Llama2-7B": 32, "Llama2-13B": 40, "Llama2-70B": 80}
minimal_channel_per_block = {"Llama2-7B": 5, "Llama2-13B": 8, "Llama2-70B": 6}
pipeline_parallel_mode_list = ["pipeline_parallel", "pipeline_parallel_embedding"]
model_parallel_mode_list = ["model_parallel", "model_parallel_embedding", "model_parallel_FC"]

def get_args():
    parser = argparse.ArgumentParser('Process model parameters.')
    parser.add_argument("--filename", help="Name of weight file")
    parser.add_argument("--model", choices=["llama-2-7b", "llama-2-13b", "llama-2-70b", "bloom"], help="model choice")
    parser.add_argument("--GEMV", choices=["reuse-GB", "reuse-bank", "no-reuse"], help="GEMV choice, inner product keeps accumulation results and re-write GB, outer product keeps GB and re-write accumulation register.", default="no-reuse")
    parser.add_argument("--reuse-size", type=int, help="reuse size for either reuse-GB or reuse-bank, depends on number of MAC register size", default=2)
    parser.add_argument("--DRAM-column", type=int, help="DRAM chip columns", default=1024)
    parser.add_argument("--DRAM-row", type=int, help="DRAM chip rows", default=1024*16)
    parser.add_argument("--burst-length", type=int, help="Burst length", default=16)
    parser.add_argument("--num-banks", type=int, help="bank number per channel", default=16)
    parser.add_argument("--num-channels", type=int, help="channel number per DIMM", default=32)
    parser.add_argument("--max-seq-len", type=int, help="maximum sequence length the model supports", default=4096)
    parser.add_argument("--threads", type=int, help="threads to use", default=16)
    parser.add_argument("--pim-memory-mapping", action="store_true", help="Simulate pim memory mapping, use naive computation")
    parser.add_argument("--pim-compute", action="store_true", help="Simulate pim computation")
    parser.add_argument("--trace-prepare", action="store_true")
    parser.add_argument("--trace-norm", action="store_true")
    parser.add_argument("--trace-fc-kqvo", action="store_true")
    parser.add_argument("--trace-attention", action="store_true")
    parser.add_argument("--trace-softmax", action="store_true")
    parser.add_argument("--trace-fc-ffn", action="store_true")
    parser.add_argument("--trace-activation", action="store_true")
    parser.add_argument("--op-trace", action="store_true", help="Print operation traces")
    parser.add_argument("--only-trace", action="store_true", help="Skip functional verification and only generate memory trace")
    parser.add_argument("--embedding", action="store_true", help="generate traces for input embedding")
    parser.add_argument("--Llama", action="store_true", help="Llama2 7B and 13B")
    parser.add_argument("--Llama-GQA", action="store_true", help="Llama2 70B and Llama3")
    parser.add_argument("--BLOOM", action="store_true", help="BLOOM")
    parser.add_argument("--OPT-66B", action="store_true", help="OPT-66B")
    parser.add_argument("--GPT3-175B", action="store_true", help="GPT-175B")
    parser.add_argument("--GPT3-175B-TP-8", action="store_true", help="GPT-175B")
    parser.add_argument("--only-FC", action="store_true", help="generate traces for aim baseline, only FC layers")
    parser.add_argument("--double-bank", action="store_true", help="one transformer block map to 2x banks")
    parser.add_argument("--quad-bank", action="store_true", help="one transformer block map to 4x banks")
    parser.add_argument("--multi-tb-per-device", action="store_true")
    parser.add_argument("--ffn_dim", type=int, help="FFN dimension")
    parser.add_argument("--n_heads", type=int, help="Number of heads")
    parser.add_argument("--n_kv_heads", type=int, help="Number of kv heads for GQA", default=8)
    parser.add_argument("--model-parallel", action="store_true", help="assign multiple transformer blocks for each CXL device")
    parser.add_argument("--pipeline-parallel", action="store_true")
    # parser.add_argument("--half-devices", action="store_true", help="use 16 devices with 256GB capacity setup")
    parser.add_argument("--channels-per-block", type=int, help="Channles required for a Transformer Block")
    # parser.add_argument("--resources", type=int, help="128 times of resources mean 7B uses 2*128=256 channels, 70B uses 512 channels")
    parser.add_argument("--FC-devices", type=int, help="devices utilized for FC layer in model parallel")
    parser.add_argument("--seqlen", type=int, help="specify seqlen for only trace mode", default=4096)
    parser.add_argument("--trace-file", help="Name of generated trace file", default="null.log")
    args = parser.parse_args()
    return args

def compare_1d(a, b, name):
    if not torch.equal(a, b):
        p1 = False
        p2 = False
        p3 = False
        p5 = False
        p10 = False
        p_large = False
        a_flatten = a.reshape(-1)
        b_flatten = b.reshape(-1)
        error_count = 0
        error_lst = []
        for i in range(a_flatten.shape[0]):
            diff = abs((a_flatten[i] - b_flatten[i]) / b_flatten[i])
            if diff > 0.001:
                error_count += 1
                error_lst.append((i, a_flatten[i].item(), b_flatten[i].item()))
                if diff > 0.1:
                    p_large = True
                elif diff > 0.05:
                    p10 = True
                elif diff > 0.03:
                    p5 = True
                elif diff > 0.02:
                    p3 = True
                elif diff > 0.01:
                    p2 = True
                else:
                    p1 = True

        if p_large:
            print(name + " has error > 10%")
        elif p10:
            print("{} has {}/{} items with 0.1% < error < {}".format(name, error_count, a_flatten.shape[0], "10%"))
        elif p5:
            print("{} has {}/{} items with 0.1% < error < {}".format(name, error_count, a_flatten.shape[0], "5%"))
        elif p3:
            print("{} has {}/{} items with 0.1% < error < {}".format(name, error_count, a_flatten.shape[0], "3%"))
        elif p2:
            print("{} has {}/{} items with 0.1% < error < {}".format(name, error_count, a_flatten.shape[0], "2%"))
        elif p1:
            print("{} has {}/{} items with 0.1% < error < {}".format(name, error_count, a_flatten.shape[0], "1%"))
        else:
            print("{} has error < 0.1%".format(name))

        for error in error_lst:
            print(error)

def compare(a, b, name):
    if torch.equal(a, b):
        print(name + " is verified")
    else:
        if debug:
            if len(a.shape) == 2:
                for i in range(a.shape[0]):
                    compare_1d(a[i], b[i], name + str(i))
            elif len(a.shape) > 2:
                print(a.shape, b.shape)
                a_tmp = a.reshape(-1, a.shape[-1] * a.shape[-2])
                b_tmp = b.reshape(-1, b.shape[-1] * b.shape[-2])
                for i in range(a_tmp.shape[0]):
                    compare_1d(a_tmp[i], b_tmp[i], name + str(i))
            else:
                print(a.shape, b.shape)
                compare_1d(a, b, name)
        else:
            compare_1d(a, b, name)
        print()
    

        

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.reshape(*shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Use rotary positional embedding to avoid the tail effects of absolute positional embedding
    starred expression is used to unpack variables
    xq = torch.Size([1, 1, 32, 128]) 
    *xq.reshape = torch.Size([1, 1, 32, 64, 2])   [a, b] in the last dimension
    xq_ = torch.Size([1, 1, 32, 64])              [a + jb] in the last dimension
    freqs_cis = torch.Size([1, 64])
    reshape_for_broadcast(freqs_cis, xq_) = torch.Size([1, 1, 1, 64])
    xq_ * freqs_cis = torch.Size([1, 1, 32, 64])
    torch.view_as_real(xq_ * freqs_cis) = torch.Size([1, 1, 32, 64, 2])
    xq_out = torch.Size([1, 1, 32, 128])
    """
    xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def RMSNorm(x, weight):
    eps = 1e-5
    x_float = x
    norm = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
    output = norm.type_as(x) * weight
    return output
