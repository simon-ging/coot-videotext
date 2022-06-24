"""
Test GPU/RAM profiling.
"""

from nntrainer.utils_torch import profile_gpu_and_ram


def test_profiling() -> None:
    """
    Test profiling.
    """
    # get the current profile values
    (gpu_names, total_memory_per, used_memory_per, load_per, ram_total, ram_used, ram_avail) = profile_gpu_and_ram()
    # average / sum over all GPUs
    gpu_mem_used: float = sum(used_memory_per)
    gpu_mem_total: float = sum(total_memory_per)
    gpu_mem_percent: float = gpu_mem_used / max(1, gpu_mem_total)
    load_avg: float = sum(load_per) / max(1, len(load_per))

    print("Metrics.PROFILE_GPU_MEM_USED", gpu_mem_used)
    print("Metrics.PROFILE_GPU_MEM_TOTAL", gpu_mem_total)
    print("Metrics.PROFILE_GPU_LOAD", load_avg)
    print("Metrics.PROFILE_RAM_USED", ram_used)
    print("Metrics.PROFILE_RAM_TOTAL", ram_total)
    print("Metrics.PROFILE_GPU_MEM_PERCENT", gpu_mem_percent)
    print("Metrics.PROFILE_RAM_AVAILABLE", ram_avail)

    # log the values
    gpu_names_str = " ".join(set(gpu_names))
    multi_load, multi_mem = "", ""
    if len(load_per) > 1:
        multi_load = " [" + ", ".join(f"{load:.0%}" for load in load_per) + "]"
        multi_mem = " [" + ", ".join(f"{mem:.1f}GB" for mem in used_memory_per) + "]"
    print(f"RAM GB used/avail/total: {ram_used:.1f}/{ram_avail:.1f}/{ram_total:.1f} - "
          f"GPU {gpu_names_str} Load: {load_avg:.1%}{multi_load} "
          f"Mem: {gpu_mem_used:.1f}GB/{gpu_mem_total:.1f}GB{multi_mem}")


if __name__ == "__main__":
    test_profiling()
