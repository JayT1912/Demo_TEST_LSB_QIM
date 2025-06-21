# 🎵 QIM + LSB: Audio Steganography System in Python 🎶

This is a Python-based audio steganography system that allows you to hide and extract messages in audio files using the **QIM** (Quantization Index Modulation) and **LSB** (Least Significant Bit) methods. The system provides encoding and decoding capabilities, allowing you to embed secret messages into audio and extract them later while ensuring minimal distortion to the original audio quality.

## 📌 Features

1. **📝 Message Encoding**:
   - Hide text messages within an audio file using the **QIM** and **LSB** methods.
   - Supports various audio formats like WAV, MP3, FLAC, and others.
   - You can adjust the encoding quality with the `Step Size` parameter to control the amount of distortion.

2. **🔓 Message Decoding**:
   - Extract hidden messages from audio files.
   - The system supports identifying whether the audio file contains a hidden message or is original.
   - Displays the decoded message after extraction.

3. **📊 Comparison and Visualization**:
   - Compare the original and modified audio files to check for differences and calculate the **Signal-to-Noise Ratio** (SNR).
   - Visualize the waveforms of both audio files for comparison.
   - Export the comparison results into a PDF report.

## 📂 Files

1. **encode.py** - This script is used to encode the secret message into the audio file.
   - It uses the **QIM+LSB** method to hide the message in the audio.
   - You can adjust the `Step Size` to control the precision of encoding.

2. **decode.py** - This script is used to decode the hidden message from the audio file.
   - It extracts the bits from the LSB of the audio samples and converts them back to text.

3. **check.py** - This script allows you to compare the original and modified audio files.
   - It calculates the **Signal-to-Noise Ratio** (SNR) to evaluate the quality of the audio.
   - Provides a graphical comparison of the waveforms of both audio files and exports the results to a PDF.

## 🛠 Installation

To use this system, you'll need Python installed along with the required libraries. You can install the necessary dependencies using `pip`:

```bash
pip install numpy soundfile matplotlib tkinter
