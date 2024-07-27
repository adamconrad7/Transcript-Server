import os
import io
from pydub import AudioSegment


def convert(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
#    dir_name = 'data/formats'
#
#    for filename in os.listdir(dir_name):
#        path = os.path.join(dir_name, filename)
#        audio = AudioSegment.from_file(path)
#
    print(f"{audio.channels} channel(s), {audio.frame_rate} Hz, {audio.sample_width * 8} bit, {audio.frame_count()} frames")
    # enforce single channel
    if audio.channels > 1:
        print('converting to mono')
        audio = audio.set_channels(1)
    
    # Convert to 16-bit PCM
    audio = audio.set_sample_width(2)
    
    # Convert to 16kHz sample rate
    audio = audio.set_frame_rate(16000)
    
    # Export as wav
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    
    return buffer.getvalue(), len(audio)/ 1000.0

#if __name__ == "__main__":
#    main()
