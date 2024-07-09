import torch
import torchaudio
import time
import json
import numpy as np
from tqdm import tqdm
from src.model import model
from src.config import config
from omegaconf import OmegaConf

def generate_random_audio(duration_seconds, sample_rate=16000):
    """Generate random noise audio."""
    num_samples = int(duration_seconds * sample_rate)
    return torch.randn(1, num_samples)

def get_gpu_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
    return 0

def run_benchmark(num_files, duration_seconds, configs):
    results = []
    for cfg in tqdm(configs, desc="Testing configurations"):
        model.change_attention_model("rel_pos_local_attn", [cfg['attn_context'], cfg['attn_context']])
        model.change_subsampling_conv_chunking_factor(cfg['subsampling_factor'])

        # Set the decoding strategy
        decoding_cfg = OmegaConf.create({
            "strategy": cfg['decoding_strategy'],
            "batch_size": cfg['batch_size'],
            "chunk_size_in_secs": cfg['chunk_size']
        })
        model.change_decoding_strategy(decoding_cfg)

        total_time = 0
        total_audio_length = num_files * duration_seconds
        max_gpu_mem_used = 0

        for _ in range(num_files):
            audio = generate_random_audio(duration_seconds)

            start_time = time.time()
            with torch.no_grad():
                _ = model.transcribe([audio.numpy().squeeze()], batch_size=cfg['batch_size'])
            end_time = time.time()

            total_time += end_time - start_time
            max_gpu_mem_used = max(max_gpu_mem_used, get_gpu_memory_usage())

        rtf = total_time / total_audio_length
        results.append({**cfg, 'rtf': rtf, 'total_time': total_time, 'gpu_mem_used': max_gpu_mem_used})

        # Force CUDA to release memory
        torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    #num_files = 5  # Number of random audio files to generate for each config
    num_files = 2  # Number of random audio files to generate for each config
    duration_seconds = 1800  # Duration of each audio file (5 minutes)
    

    configs = [
        {'attn_context': 128, 'subsampling_factor': 1, 'batch_size': 4, 'decoding_strategy': 'greedy_batch', 'chunk_size': 20},
        {'attn_context': 128, 'subsampling_factor': 1, 'batch_size': 40, 'decoding_strategy': 'greedy_batch', 'chunk_size': 20},
        {'attn_context': 128, 'subsampling_factor': 1, 'batch_size': 4, 'decoding_strategy': 'greedy_batch', 'chunk_size': 40},
        {'attn_context': 128, 'subsampling_factor': 1, 'batch_size': 40, 'decoding_strategy': 'greedy_batch', 'chunk_size': 40},
        {'attn_context': 128, 'subsampling_factor': 1, 'batch_size': 4, 'decoding_strategy': 'greedy_batch', 'chunk_size': 80},
        {'attn_context': 128, 'subsampling_factor': 1, 'batch_size': 40, 'decoding_strategy': 'greedy_batch', 'chunk_size': 80},
    ]

    results = run_benchmark(num_files, duration_seconds, configs)
    
    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print best configuration
    best_config = min(results, key=lambda x: x['rtf'])
    print(f"Best configuration: {best_config}")

    # Print all results sorted by RTF
    print("\nAll configurations sorted by RTF:")
    for cfg in sorted(results, key=lambda x: x['rtf']):
        print(f"RTF: {cfg['rtf']:.4f}, GPU Memory: {cfg['gpu_mem_used']:.2f} GB, Config: {cfg}")
