# Artifact of the CENT Paper, ASPLOS 2025

This repository provides the following artifact required for the evaluation of **CENT**, "PIM is All You Need: A CXL-Enabled GPU-Free System for LLM Inference" paper published in ASPLOS 2025.

## Dependencies

AiM Simulator is tested and verified with `g++-11/12/13` and `clang++-15`.
Python infrastructure requires `pandas`, `matplotlib`, `torch`, and `scipy` packages.

## Build

Clone the repository recursively:

```bash
git clone --recursive https://github.com/Yufeng98/CENT.git
cd CENT
```

Install the Python packages locally or create a conda environment:

```bash
conda create -n cent python=3.10 -y
conda activate cent
pip install -r requirements.txt
```

Build the AiM simulator:

```bash
# use g++-11/12/13. e.g. (export CXX=/usr/bin/g++-12)
cd aim_simulator
mkdir build
cd build
cmake ..
make -j4
cd ../../
```

## Artifact Scripts

Remove old results:

```bash
bash remove_old_results.sh
```

### Run Simulation

Perform the end-to-end trace generation, performance simulation, and power modeling:

```bash
cd cent_simulation
# bash simulation.sh <set threads based on your platform> <sequence length gap>
bash simulation.sh 8 128
bash process_results.sh
cd ..
```

**Note:** Setting `sequence length gap` to 1 will start full simulation, generating token by token from 1 to 4096. Setting `sequence length gap` to 128 will generate tokens at index of 128, 256, ..., 4096. The final results are averaged on various sequence lengths. We show results using `sequence length gap = 1`, but full simulation takes long time and ~100GB disk space. For example, using 8 threads on a desktop takes ~8GB memory and ~24 hours for full simulation, and using 96 threads on a server takes ~64GB memory and ~12 hours for full simulation. For a quick verification, use `sequence length gap = 128`, which only takes a few hours and has minor difference in results.

Generate all figures using the following script or generate them one by one using scripts in sections below:

```bash
bash generate_figures.sh
```

After completing this step, CSV and PDF files for Figures 11–14 will be generated in the `figures` and `figure_source_data` directories. The figures in the published paper were created using `cent_simulation/Figures.xlsx` Excel file.
To reproduce the figures in the same format as those in the paper, copy the content of the generated CSV files into the `Data` sheet of the Excel file. The figures will then be generated in the `Figures` sheet.

### Figure 12

The CXL controller costs are broken down into die, packaging and Non Recurring Engineering (NRE) components. The die cost is derived from the wafer cost, considering the CXL controller die area and yield rate. The cost of 2D packaging is assumed to be 29% of chip cost (die and package). The NRE cost is influenced by chip production volumes.

```bash
python figure_scripts/figure_12.py
```

### Figure 13

CENT speedup over GPU baselines. (a) Batch = 1 Latency comparison. (b) Throughput comparison under the highest batch size that CENT and GPU and achieve. (c) TCO normalized throughput comprison.

```bash
python figure_scripts/figure_13a.py
python figure_scripts/figure_13b.py
python figure_scripts/figure_13c.py
```

### Figure 14

Analysis on Llama2-70B. (a) CENT achieves higher decoding throughputs with long context windows and 3.5K decoding sizes. (b) QoS analysis: CENT shows less query latency when achieving the similar to GPUs. (c) CENT latency breakdown with different parallelism strategies. (d) Prefill (In) and decoding (Out) latency comparison with different In/Out sizes, at maximum supported batches for both GPU and CENT.

```bash
python figure_scripts/figure_14a.py
python figure_scripts/figure_14b.py
python figure_scripts/figure_14c.py
python figure_scripts/figure_14d.py
```

### Figure 15

Power and energy analysis. (a) Power consumption, (b) GPU SM frequency and board power, and (c) energy efficiency of CENT and GPU for different stages of Llama2 models using the maximum batch size, 512 prefill tokens and 3584 decoding tokens.

```bash
python figure_scripts/figure_15a.py
python figure_scripts/figure_15c.py
```

## Citation

If you use *CENT*, please cite this paper:

> Yufeng Gu, Alireza Khadem, Sumanth Umesh, Ning Liang, Xavier Servot, Onur Mutlu, Ravi Iyer, and Reetuparna Das.
> *PIM is All You Need: A CXL-Enabled GPU-Free System for LLM Inference*,
> In 2025 International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)

```
@inproceedings{cent,
  title={PIM is All You Need: A CXL-Enabled GPU-Free System for LLM Inference},
  author={Gu, Yufeng and Khadem, Alireza and Umesh, Sumanth, and Liang, Ning and Servot, Xavier and Mutlu, Onur and Iyer, Ravi and and Das, Reetuparna},
  booktitle={2025 International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)}, 
  year={2025}
}
```

## Issues and bug reporting

We appreciate any feedback and suggestions from the community.
Feel free to raise an issue or submit a pull request on Github.
For assistance in using CENT, please contact: Yufeng Gu (yufenggu@umich.edu) and Alireza Khadem (arkhadem@umich.edu)

## Licensing

This repository is available under a [MIT license](/LICENSE).

## Acknowledgement

This work was supported in part by the NSF under the CAREER-1652294 and NSF-1908601 awards and by Intel gift.
