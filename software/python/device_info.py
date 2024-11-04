import psutil
import platform
import torch

def get_system_info():
    cpu_info = {
        "physical_cores": psutil.cpu_count(logical=False),
        "total_cores": psutil.cpu_count(logical=True),
        "max_frequency": psutil.cpu_freq().max,
        "min_frequency": psutil.cpu_freq().min,
        "current_frequency": psutil.cpu_freq().current,
        "cpu_usage": psutil.cpu_percent(interval=1),
        "cpu_model": platform.processor()
    }
    
    ram = psutil.virtual_memory()
    ram_info = {
        "total": ram.total,
        "available": ram.available,
        "percent": ram.percent
    }
    
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_cached": torch.cuda.memory_reserved(0)
        }
    
    return {
        "cpu": cpu_info,
        "ram": ram_info,
        "gpu": gpu_info
    }

def print_system_info(system_info):
    print('----------------------------------------')
    print('System Information:')
    print(f"CPU: {system_info['cpu']['cpu_model']}")
    print(f"Physical cores: {system_info['cpu']['physical_cores']}")
    print(f"Total cores: {system_info['cpu']['total_cores']}")
    print(f"Max CPU frequency: {system_info['cpu']['max_frequency']} MHz")
    print(f"Current CPU frequency: {system_info['cpu']['current_frequency']} MHz")
    print(f"CPU Usage: {system_info['cpu']['cpu_usage']}%")
    print(f"RAM: Total {system_info['ram']['total'] / (1024**3):.2f} GB, "
          f"Available {system_info['ram']['available'] / (1024**3):.2f} GB, "
          f"Usage {system_info['ram']['percent']}%")
    
    if system_info['gpu']:
        print(f"GPU: {system_info['gpu']['name']}")
        print(f"GPU Memory: Total {system_info['gpu']['memory_total'] / (1024**3):.2f} GB, "
              f"Allocated {system_info['gpu']['memory_allocated'] / (1024**3):.2f} GB, "
              f"Cached {system_info['gpu']['memory_cached'] / (1024**3):.2f} GB")
    else:
        print("GPU: Not available")
    print('----------------------------------------')

print_system_info(get_system_info())
