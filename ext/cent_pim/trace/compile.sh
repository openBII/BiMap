for f in trace*.log
do
	echo "Processing $f file...";
	tail -n 10000 $f &> logs.txt;
	cat logs.txt | grep "memory_system_cycles";
	cat logs.txt | grep "idle_cycles";
	cat logs.txt | grep "precharged_cycles";
	cat logs.txt | grep "active_cycles";
	cat logs.txt | grep "num_ACT_commands";
    cat logs.txt | grep "num_PREA_commands";
    cat logs.txt | grep "num_PRE_commands";
    cat logs.txt | grep "num_RD_commands";
    cat logs.txt | grep "num_WR_commands";
    cat logs.txt | grep "num_RDA_commands";
    cat logs.txt | grep "num_WRA_commands";
    cat logs.txt | grep "num_REFab_commands";
    cat logs.txt | grep "num_REFpb_commands";
    cat logs.txt | grep "num_ACT4_commands";
    cat logs.txt | grep "num_ACT16_commands";
    cat logs.txt | grep "num_PRE4_commands";
    cat logs.txt | grep "num_MAC_commands";
    cat logs.txt | grep "num_MAC16_commands";
    cat logs.txt | grep "num_AF16_commands";
    cat logs.txt | grep "num_EWMUL16_commands";
    cat logs.txt | grep "num_RDCP_commands";
    cat logs.txt | grep "num_WRCP_commands";
    cat logs.txt | grep "num_WRGB_commands";
    cat logs.txt | grep "num_RDMAC16_commands";
    cat logs.txt | grep "num_RDAF16_commands";
    cat logs.txt | grep "num_WRMAC16_commands";
    cat logs.txt | grep "num_WRA16_commands";
    cat logs.txt | grep "num_TMOD_commands";
    cat logs.txt | grep "num_SYNC_commands";
done
