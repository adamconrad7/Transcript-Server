#from src.AudioProcessor import AudioProcessor
#from src.model import model  # Assuming this is how your model is imported
#
## For production
##processor = AudioProcessor(model, batch_size=8)
##transcriptions = processor.process_files('path/to/audio/files')
#
## For evaluation
#eval_processor = AudioProcessor(model, batch_size=8, eval_mode=True)
#eval_transcriptions = eval_processor.process_files('LibriSpeech/test-clean')
import os
import glob
from src.inference import transcribe_audio
import jiwer
import time
from tqdm import tqdm
import soundfile as sf

def process_librispeech(root_dir='LibriSpeech/test-clean'):
    all_ground_truths = []
    all_transcriptions = []
    total_audio_duration = 0
    total_transcription_time = 0
    file_count = 0
    start_time = time.time()

    #for speaker_dir in glob.glob(os.path.join(root_dir, '*'))[:1]:
    for speaker_dir in glob.glob(os.path.join(root_dir, '*')):
        for chapter_dir in glob.glob(os.path.join(speaker_dir, '*')):
            # Read the ground truth transcriptions
            trans_file = os.path.join(chapter_dir, f'{os.path.basename(chapter_dir.split("/")[-2] + "-" + chapter_dir.split("/")[-1])}.trans.txt')
            print(trans_file)
            with open(trans_file, 'r') as f:
                ground_truth = {line.split()[0]: ' '.join(line.split()[1:]) for line in f}
            
            # Process each audio file
            for audio_file in glob.glob(os.path.join(chapter_dir, '*.flac')):
                file_id = os.path.splitext(os.path.basename(audio_file))[0]

#                real_start_time = time.time()
                
                # Get audio duration
                audio_info = sf.info(audio_file)
                audio_duration = audio_info.duration
                total_audio_duration += audio_duration

                # Read audio file
                with open(audio_file, 'rb') as f:
                    audio_data = f.read()
                
                # Transcribe audio
                transcription_start = time.time()
                transcription = transcribe_audio(audio_data)
                transcription_end = time.time()

#                real_end_time = time.time()
                
                transcription_time = transcription_end - transcription_start
#                real_time = real_end_time - real_start_time

                total_transcription_time += transcription_time
#                total_real_time += real_time

                file_count += 1

#                rtf_net = transcription_time / audio_duration
#                rtf_real = real_time / audio_duration

                # Compare with ground truth
#               print(f'File: {file_id}')
#               print(f'Ground Truth: {ground_truth[file_id].lower()}')
#               print(f'Transcription: {transcription}')
#               print(f'Audio Duration: {audio_duration:.2f} seconds')
#               print(f'Transcription Time: {transcription_time:.4f} seconds')
#               print(f'RTF: {rtf:.4f}')
#               print('---')
#                print('---')
                all_ground_truths.append(ground_truth[file_id].lower())
                all_transcriptions.append(transcription)

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate WER for all transcriptions
    wer = jiwer.wer(all_ground_truths, all_transcriptions)

    # Calculate overall RTF
    #overall_rtf = total_transcription_time / total_audio_duration
    overall_rtf_net = total_transcription_time / total_audio_duration
    overall_rtf_real = total_time / total_audio_duration

    print("\nPerformance Metrics:")
    print(f'Total Execution Time: {total_time:.2f} seconds')
    print(f'Total Audio Duration: {total_audio_duration:.2f} seconds')
    print(f'Total Net Transcription Time: {total_transcription_time:.2f} seconds')
    print(f'Average Net Transcription Time per File: {total_transcription_time/file_count:.4f} seconds')
    print(f'Average Real Processing Time per File: {total_time/file_count:.4f} seconds')
    print(f'Files Processed: {file_count}')
    print(f'Overall Net RTF: {overall_rtf_net:.4f}')
    print(f'Overall Real RTF: {overall_rtf_real:.4f}')
    print(f'Overall WER: {wer:.4f}')

#    print("\nPerformance Metrics:")
#    print(f'Total Execution Time: {total_time:.2f} seconds')
#    print(f'Total Audio Duration: {total_audio_duration:.2f} seconds')
#    print(f'Total Transcription Time: {total_transcription_time:.2f} seconds')
#    print(f'Average Transcription Time per File: {total_transcription_time/file_count:.4f} seconds')
#    print(f'Files Processed: {file_count}')
#    print(f'Overall RTF: {overall_rtf:.4f}')
#    print(f'Overall WER: {wer:.4f}')

if __name__ == '__main__':
    process_librispeech()
