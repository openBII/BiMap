trace_file=./unit/gen_only.txt
./../ext/cent_pim/aim_simulator/build/ramulator2 -f ./../ext/cent_pim/aim_simulator/test/example.yaml -t $trace_file > $trace_file.log
trace_file=./unit/ffn_only.txt
./../ext/cent_pim/aim_simulator/build/ramulator2 -f ./../ext/cent_pim/aim_simulator/test/example.yaml -t $trace_file > $trace_file.log
trace_file=./unit/gen_only_mac.txt
./../ext/cent_pim/aim_simulator/build/ramulator2 -f ./../ext/cent_pim/aim_simulator/test/example.yaml -t $trace_file > $trace_file.log
trace_file=./unit/gen_only_mac_abk.txt
./../ext/cent_pim/aim_simulator/build/ramulator2 -f ./../ext/cent_pim/aim_simulator/test/example.yaml -t $trace_file > $trace_file.log
trace_file=./unit/gen_only_rd.txt
./../ext/cent_pim/aim_simulator/build/ramulator2 -f ./../ext/cent_pim/aim_simulator/test/example.yaml -t $trace_file > $trace_file.log