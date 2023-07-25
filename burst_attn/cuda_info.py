import pynvml

pynvml.nvmlInit()

def bits_to_mib(bits):
    mib = bits / (1024 * 1024)
    return mib

def getMemoryTotal():
    count = pynvml.nvmlDeviceGetCount()
    total_mib=0
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mib += bits_to_mib(meminfo.used)
    return total_mib



def getMemory(device_index):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(meminfo.used)

if __name__ == "__main__":
    pynvml.nvmlInit()
    print(f"{getMemoryTotal()}Mib")