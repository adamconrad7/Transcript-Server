# Effecient Audio Transcription
## Introduction
This project aims to address the gap between transcription technology at the bleeding edge and usable, performant implementations of these technologies. 

## Usage
Currently, your best bet is to clone the repo on a machine with Nvidia drivers and Doxker. Use Docker to build and start the container. You will need considrable (50+ GB) space to build the image due to Nvidia tooling. Docker will start a local server on the instance that responds to POST requests with your file. 

## Benchmarks
All evals done on LibriSpeech test-clean
### AWS g4dn.xlarge (Nvidia T4) 16GB/4vCPUs
#### Stock parakeet-ctc-0.6b:

Performance Metrics:
Total Execution Time: 331.72 seconds
Total Audio Duration: 19452.48 seconds
Total Transcription Time: 326.83 seconds
Average Transcription Time per File: 0.1247 seconds                                
Files Processed: 2620
Overall RTF: 0.0168 
Overall WER: 0.0204

Overall RTF: 0.0168 * $.526 /hour (g4dn.xlarge) = .0088
### To Do
- Explain project
- Cost analysis of existing transcription services
- Optimize data loading
- Web-reachable server
- Optimize model for inference
    - tensorRT conversion
    - mixed precision
- Automate provisioning
- Check for local model instead of Dl

