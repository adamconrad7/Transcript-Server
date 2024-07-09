import os
import glob
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import soundfile as sf
import jiwer

class AudioProcessor:
    def __init__(self, model, batch_size=8, eval_mode=False):
        self.model = model
        self.batch_size = batch_size
        self.eval_mode = eval_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    class AudioDataset(Dataset):
        def __init__(self, audio_files, ground_truths=None):
            self.audio_files = audio_files
            self.ground_truths = ground_truths

        def __len__(self):
            return len(self.audio_files)

        def __getitem__(self, idx):
            with open(self.audio_files[idx], 'rb') as f:
                audio_data = f.read()
            if self.ground_truths:
                return audio_data, self.ground_truths[idx]
            return audio_data

    def batch_transcribe(self, batch):
        with torch.no_grad(), torch.cuda.amp.autocast():
            if self.eval_mode:
                audios = [item[0] for item in batch]
            else:
                audios = batch
            transcriptions = self.model.transcribe(audios)
        return transcriptions

    def process_files(self, root_dir):
        audio_files = glob.glob(os.path.join(root_dir, '**', '*.flac'), recursive=True)
        
        if self.eval_mode:
            ground_truths = self._load_ground_truths(root_dir)
            dataset = self.AudioDataset(audio_files, ground_truths)
        else:
            dataset = self.AudioDataset(audio_files)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)

        total_audio_duration = 0
        total_processing_time = 0
        all_transcriptions = []
        all_ground_truths = []

        start_time = time.time()

        for batch in tqdm(dataloader):
            batch_start_time = time.time()
            if self.eval_mode:
                audios, truths = batch
                transcriptions = self.batch_transcribe(audios)
                all_ground_truths.extend(truths)
            else:
                transcriptions = self.batch_transcribe(batch)
            batch_end_time = time.time()

            all_transcriptions.extend(transcriptions)
            
            batch_duration = sum([sf.info(audio_file).duration for audio_file in (audios if self.eval_mode else batch)])
            total_audio_duration += batch_duration
            total_processing_time += batch_end_time - batch_start_time

        end_time = time.time()
        total_time = end_time - start_time

        self._print_metrics(total_time, total_audio_duration, total_processing_time, len(audio_files))

        if self.eval_mode:
            wer = jiwer.wer(all_ground_truths, all_transcriptions)
            print(f'Word Error Rate (WER): {wer:.4f}')

        return all_transcriptions

    def _load_ground_truths(self, root_dir):
        ground_truths = {}
        for trans_file in glob.glob(os.path.join(root_dir, '**', '*.trans.txt'), recursive=True):
            with open(trans_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        file_id, text = parts
                        ground_truths[file_id] = text.lower()
        return ground_truths

    def _print_metrics(self, total_time, total_audio_duration, total_processing_time, num_files):
        rtf = total_processing_time / total_audio_duration
        throughput = num_files / total_time

        print("\nPerformance Metrics:")
        print(f'Total Execution Time: {total_time:.2f} seconds')
        print(f'Total Audio Duration: {total_audio_duration:.2f} seconds')
        print(f'Total Processing Time: {total_processing_time:.2f} seconds')
        print(f'Files Processed: {num_files}')
        print(f'Real-Time Factor (RTF): {rtf:.4f}')
        print(f'Throughput: {throughput:.2f} files/second')

