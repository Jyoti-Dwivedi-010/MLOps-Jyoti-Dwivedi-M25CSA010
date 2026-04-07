#!/bin/bash
# Cleanup GPU memory after training
# Usage: ./scripts/cleanup_gpu.sh

echo "Cleaning up GPU memory..."

python -c "
import torch
import gc

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    # Print memory stats
    device = torch.device('cuda')
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    
    print(f'GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB')
    print('GPU memory cleanup complete.')
else:
    print('No CUDA device available.')
"

echo "Done."
