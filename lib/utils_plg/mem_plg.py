
import gc
import torch

def memory_stats(idx):
    print("Mem state ", idx, torch.cuda.memory_allocated()/1024**2, torch.cuda.memory_reserved()/1024**2)

def collect_cache():
    gc.collect()
    torch.cuda.empty_cache()
