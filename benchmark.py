import os
import json
import time
import random
import argparse
import requests
from jiwer import wer
from pydub import AudioSegment
from tqdm import tqdm

def load_transcriptions(trans_file):
    transcriptions = {}
    with open(trans_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                file_id, text = parts
                transcriptions[file_id] = text
    return transcriptions

def transcribe_file(file_path, worker_url):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(worker_url, files=files)
    return response.json()['transcription'] if response.status_code == 200 else None

def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path, format="flac")
    return len(audio) / 1000.0  # Convert milliseconds to seconds

def benchmark_dataset(dataset_path, worker_url, percentage):
    all_flac_files = []
    for root, _, files in os.walk(dataset_path):
        all_flac_files.extend([os.path.join(root, f) for f in files if f.endswith('.flac')])
    
    # Randomly select a percentage of files
    num_files_to_test = max(1, int(len(all_flac_files) * percentage / 100))
    files_to_test = random.sample(all_flac_files, num_files_to_test)
    
    results = []
    total_time = 0
    total_audio_duration = 0
    
    # Create a progress bar
    pbar = tqdm(total=len(files_to_test), desc="Processing files")
    
    for file_path in files_to_test:
        file = os.path.basename(file_path)
        file_id = os.path.splitext(file)[0]
        
        start_time = time.time()
        hypothesis = transcribe_file(file_path, worker_url)
        end_time = time.time()
        
        if hypothesis is not None:
            trans_file = os.path.join(os.path.dirname(file_path), f"{file_id.rsplit('-', 1)[0]}.trans.txt")
            reference_transcriptions = load_transcriptions(trans_file)
            reference = reference_transcriptions.get(file_id, '').lower()
            error_rate = wer(reference, hypothesis.lower())
            
            processing_time = end_time - start_time
            total_time += processing_time
            
            audio_duration = get_audio_duration(file_path)
            total_audio_duration += audio_duration
            
            results.append({
                'file': file,
                'reference': reference,
                'hypothesis': hypothesis,
                'wer': error_rate,
                'processing_time': processing_time,
                'audio_duration': audio_duration
            })
        
        # Update the progress bar
        pbar.update(1)
    
    # Close the progress bar
    pbar.close()
    
    avg_wer = sum(r['wer'] for r in results) / len(results) if results else 0
    rtf = total_time / total_audio_duration if total_audio_duration > 0 else 0
    
    return {
        'results': results,
        'average_wer': avg_wer,
        'total_time': total_time,
        'total_audio_duration': total_audio_duration,
        'rtf': rtf,
        'files_tested': len(files_to_test),
        'total_files': len(all_flac_files)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark LibriSpeech dataset')
    parser.add_argument('--dataset_path', type=str, default='../data/LibriSpeech/test-clean',
                        help='Path to the LibriSpeech dataset')
    parser.add_argument('--percentage', type=float, default=100.0,
                        help='Percentage of the dataset to test (default: 100.0)')
    args = parser.parse_args()

    aws_ip = os.environ.get('AWS_IP')
    if not aws_ip:
        raise ValueError("AWS_IP environment variable is not set")
    
    worker_url = f'http://{aws_ip}:8000/transcribe'
    
    print(f"Worker URL: {worker_url}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Testing {args.percentage}% of the dataset")
    
    benchmark_results = benchmark_dataset(args.dataset_path, worker_url, args.percentage)
    
    print(f"\nFiles tested: {benchmark_results['files_tested']} out of {benchmark_results['total_files']}")
    print(f"Average WER: {benchmark_results['average_wer']:.4f}")
    print(f"Total processing time: {benchmark_results['total_time']:.2f} seconds")
    print(f"Total audio duration: {benchmark_results['total_audio_duration']:.2f} seconds")
    print(f"Real-time factor: {benchmark_results['rtf']:.4f}")
    
    # Save detailed results to a file
#    with open('benchmark_results.json', 'w') as f:
#        json.dump(benchmark_results, f, indent=2)
