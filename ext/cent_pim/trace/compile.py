import sys
file = open(sys.argv[1], 'r')
lines = file.readlines()
log = ""
total_cycles = -1
total_idle_cycles = -1
total_active_cycles = -1
total_precharged_cycles = -1
commands = ["ACT",
            "PREA",
            "PRE",
            "RD",
            "WR",
            "RDA",
            "WRA",
            "REFab",
            "REFpb",
            "ACT4",
            "ACT16",
            "PRE4",
            "MAC",
            "MAC16",
            "AF16",
            "EWMUL16",
            "RDCP",
            "WRCP",
            "WRGB",
            "RDMAC16",
            "RDAF16",
            "WRMAC16",
            "WRA16",
            "TMOD",
            "SYNC",
            "EOC"]
command_count = {}
for command in commands:
    command_count[command] = 0
for line in lines:
    if "Processing" in line:
        if total_cycles != -1:
            print(f"{log}\t{total_cycles/2000000.000}\t{total_active_cycles/32.00/2000000.000}\t{total_precharged_cycles/32.00/2000000.000}\t{100.00-(total_idle_cycles/32.00/total_cycles)*100.00}\t", end="")
            for command in commands:
                print(f"{command_count[command]}\t", end="")
                command_count[command] = 0
            print()
            total_cycles = -1
            total_idle_cycles = -1
            total_active_cycles = -1
            total_precharged_cycles = -1
        log = line.split()[1]
    if "memory_system_cycles" in line:
        total_cycles = int(line.split()[1])
    if "idle_cycles" in line:
        if total_idle_cycles == -1:
            total_idle_cycles = int(line.split()[1])
        else:
            total_idle_cycles += int(line.split()[1])
    if "active_cycles" in line:
        if total_active_cycles == -1:
            total_active_cycles = int(line.split()[1])
        else:
            total_active_cycles += int(line.split()[1])
    if "precharged_cycles" in line:
        if total_precharged_cycles == -1:
            total_precharged_cycles = int(line.split()[1])
        else:
            total_precharged_cycles += int(line.split()[1])
    for command in commands:
        if "num_" + command + "_commands" in line:
            command_count[command] += int(line.split()[1])
if total_cycles != -1:
    print(f"{log}\t{total_cycles/2000000.000}\t{total_active_cycles/32.00/2000000.000}\t{total_precharged_cycles/32.00/2000000.000}\t{100.00-(total_idle_cycles/32.00/total_cycles)*100.00}\t", end="")
    for command in commands:
        print(f"{command_count[command]}\t", end="")
    print()