
def load_QoS_file(filename):
    dic = {}
    latency = []
    throughput = []
    batch = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            lst = line.split(",")
            batch.append(float(lst[0]))
            latency.append(float(lst[1])/60)
            throughput.append(float(lst[2]))
    dic["batch"] = batch
    dic["latency"] = latency
    dic["throughput"] = throughput
    return dic