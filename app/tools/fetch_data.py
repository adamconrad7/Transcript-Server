import urllib.request
import tarfile
import os

url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
filename = "test-clean.tar.gz"

print("Downloading LibriSpeech test-clean...")
urllib.request.urlretrieve(url, filename)

print("Extracting files...")
with tarfile.open(filename, "r:gz") as tar:
    tar.extractall()

print("Cleaning up...")
os.remove(filename)

print("Download and extraction complete.")
