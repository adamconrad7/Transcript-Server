import torch
import time
import json
import numpy as np
from tqdm import tqdm
from src.model import model
from src.config import config
import subprocess
from omegaconf import OmegaConf
import threading

def generate_random_audio(duration_seconds, sample_rate=16000):
    """Generate random noise audio."""
    num_samples = int(duration_seconds * sample_rate)
    return torch.randn(1, num_samples)

def get_gpu_memory_map():
    """Get the current GPU memory usage."""
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory[0]  # Assuming we're using the first GPU

def monitor_gpu_memory(stop_event, memory_usage):
    while not stop_event.is_set():
        memory_usage.append(get_gpu_memory_map())
        time.sleep(0.1)  # Check every 100ms

def process_audio_chunked(audio, chunk_duration=200, sample_rate=16000):
    """Process audio in chunks."""
    chunk_size = chunk_duration * sample_rate
    num_chunks = len(audio) // chunk_size + (1 if len(audio) % chunk_size != 0 else 0)
    
    results = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(audio))
        chunk = audio[start:end]
        
        with torch.no_grad():
            chunk_result = model.transcribe([chunk], batch_size=1)
        results.append(chunk_result[0])
        
        torch.cuda.empty_cache()  # Clear GPU cache after each chunk
    
    return " ".join(results)

def run_benchmark(durations, cfg, use_chunking=False):
    results = []
    model.change_attention_model("rel_pos_local_attn", [cfg['attn_context'], cfg['attn_context']])
    model.change_subsampling_conv_chunking_factor(cfg['subsampling_factor'])
    
    decoding_cfg = OmegaConf.create({
        "strategy": cfg['decoding_strategy'],
        "batch_size": cfg['batch_size'],
        "chunk_size_in_secs": cfg['chunk_size']
    })
    model.change_decoding_strategy(decoding_cfg)

    for duration in tqdm(durations, desc="Testing durations"):
        audio = generate_random_audio(duration)
        
        # Start memory monitoring
        stop_event = threading.Event()
        memory_usage = []
        monitor_thread = threading.Thread(target=monitor_gpu_memory, args=(stop_event, memory_usage))
        monitor_thread.start()

        start_time = time.time()
        with torch.no_grad():
            if use_chunking:
                _ = process_audio_chunked(audio.numpy().squeeze())
            else:
                _ = model.transcribe([audio.numpy().squeeze()], batch_size=cfg['batch_size'])
        end_time = time.time()
        
        # Stop memory monitoring
        stop_event.set()
        monitor_thread.join()

        total_time = end_time - start_time
        rtf = total_time / duration
        
        results.append({
            'duration': duration,
            'rtf': rtf,
            'total_time': total_time,
            'initial_gpu_mem': memory_usage[0],
            'peak_gpu_mem': max(memory_usage),
            'gpu_mem_increase': max(memory_usage) - memory_usage[0],
            'use_chunking': use_chunking
        })
        
        torch.cuda.empty_cache()
        time.sleep(1)  # Give some time for memory to be freed

    return results

if __name__ == "__main__":
    durations = [30, 300, 600, 1800, 3600, 7200]  # Test durations in seconds
    #durations = [30, 300, 600, 1800]  # Test durations in seconds
    
    best_config = {
        'attn_context': 128,
        'subsampling_factor': 1,
        'batch_size': 40,
        'decoding_strategy': 'greedy_batch',
        'chunk_size': 20
    }
    
    # Run benchmark with chunking
    results_with_chunking = run_benchmark(durations, best_config, use_chunking=True)
    
    print("\nAll results sorted by duration:")
    for result in sorted(results_with_chunking, key=lambda x: x['duration']):
        print(f"Duration: {result['duration']}s, Method: With Chunking")
        print(f"  RTF: {result['rtf']:.4f}, Total Time: {result['total_time']:.2f}s")
        print(f"  Initial GPU Mem: {result['initial_gpu_mem']} MB")
        print(f"  Peak GPU Mem: {result['peak_gpu_mem']} MB")
        print(f"  GPU Mem Increase: {result['gpu_mem_increase']} MB")
        print()
    
