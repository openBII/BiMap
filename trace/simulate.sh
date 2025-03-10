seq_len=9216
trace_file=./32_channels_per_device/pipeline_parallel/Llama2-7B/trace_32_channels_per_block_seqlen_$seq_len.txt
./../ext/cent_pim/aim_simulator/build/ramulator2 -f ./../ext/cent_pim/aim_simulator/test/example.yaml -t $trace_file > $trace_file.log