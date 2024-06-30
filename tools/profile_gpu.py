import nemo.collections.asr as nemo_asr
import torch
import torchaudio

class ParakeetBenchmark:
    def __init__(self, model_name="nvidia/parakeet-ctc-0.6b"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        print(f"Loading model: {model_name}")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Model loaded successfully")

    def generate_audio(self, duration):
        sample_rate = 16000
        t = torch.linspace(0, duration, int(duration * sample_rate))
        audio = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # 440 Hz sine wave
        return audio.to(self.device), sample_rate

    def preprocess_audio(self, audio, sample_rate):
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(self.device)
            audio = resampler(audio)
        return audio

    def run_inference(self, audio):
        with torch.no_grad():
            logprobs, logprobs_len = self.model.forward(input_signal=audio, input_signal_length=torch.tensor([audio.shape[1]]).to(self.device))
            return self.model.decoding.ctc_decoder_predictions_tensor(logprobs)[0]

    def measure_performance(self, duration):
        audio, sample_rate = self.generate_audio(duration)
        audio = self.preprocess_audio(audio, sample_rate)

        start_time = time.time()
        start_mem = torch.cuda.memory_allocated()

        transcription = self.run_inference(audio)

        end_time = time.time()
        end_mem = torch.cuda.memory_allocated()

        processing_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated() - start_mem
        rtf = processing_time / duration

        return {
            "duration": duration,
            "processing_time": processing_time,
            "rtf": rtf,
            "peak_memory": peak_memory,
            "transcription_length": len(transcription)
        }

    def run_benchmark(self, durations):
        results = []
        for duration in durations:
            try:
                result = self.measure_performance(duration)
                results.append(result)
                print(f"Processed {duration}s audio: RTF={result['rtf']:.4f}, Peak Memory={result['peak_memory']/1e6:.2f}MB")
            except RuntimeError as e:
                print(f"Error processing {duration}s audio: {str(e)}")
                break
        return results

# Usage
if __name__ == "__main__":
    benchmark = ParakeetBenchmark()
    durations = [60, 300, 600, 900, 1200, 1500, 1800]  # 1 min to 30 mins
    results = benchmark.run_benchmark(durations)
    # Add code here to log or print the results as needed









































































