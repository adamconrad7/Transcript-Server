import os
from pydub import AudioSegment


def main():
    dir_name = 'data/formats'

    for filename in os.listdir(dir_name):
        path = os.path.join(dir_name, filename)
        audio = AudioSegment.from_file(path)

        print(f"{audio.channels} channel(s), {audio.frame_rate} Hz, {audio.sample_width * 8} bit, {audio.frame_count()} frames")

if __name__ == "__main__":
    main()
