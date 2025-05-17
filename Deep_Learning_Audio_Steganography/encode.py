import torch
import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pywt

# Định nghĩa mô hình Autoencoder
class StegoAutoencoder(nn.Module):
    def __init__(self):
        super(StegoAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Sử dụng Sigmoid để đưa giá trị âm thanh trong phạm vi [0, 1]
        )

    def forward(self, audio, message):
        x = torch.cat((audio, message), dim=1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def prepare_message(message, audio_length):
    """Chuẩn bị message dưới dạng tín hiệu sóng âm"""
    message_array = np.array([ord(c) / 256.0 for c in message], dtype=np.float32)  # Chuyển thành giá trị sóng
    if len(message_array) < audio_length:
        message_array = np.tile(message_array, audio_length // len(message_array) + 1)[:audio_length]
    else:
        message_array = message_array[:audio_length]
    return torch.tensor(message_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def wavelet_denoising(signal, wavelet="db1", level=1):
    """Denoising sử dụng Wavelet transform"""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745
    coeffs = [pywt.threshold(i, threshold, mode="soft") for i in coeffs]
    return pywt.waverec(coeffs, wavelet)

def calculate_snr(original, stego):
    """Tính Signal-to-Noise Ratio (SNR)"""
    noise = original - stego
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def encode_audio(input_audio, secret_message, model_path="autoencoder.pth"):
    # Tần số mẫu (sample rate)
    sample_rate = 16000

    device = torch.device("cpu")

    # Kiểm tra file tồn tại và định dạng
    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"File '{input_audio}' không tồn tại.")
    if not input_audio.lower().endswith('.wav'):
        raise ValueError(f"File '{input_audio}' phải có định dạng WAV.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File mô hình '{model_path}' không tồn tại.")

    try:
        # Tải audio
        audio, sr = sf.read(input_audio)
        if len(audio.shape) > 1:  # Chuyển stereo thành mono nếu cần
            audio = np.mean(audio, axis=1)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        audio = audio / np.max(np.abs(audio))  # Chuẩn hóa
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Chuẩn bị message
        message_tensor = prepare_message(secret_message, audio.shape[0])

        # Khởi tạo và tải mô hình
        model = StegoAutoencoder().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Chuyển dữ liệu về thiết bị CPU
        audio_tensor = audio_tensor.to(device)
        message_tensor = message_tensor.to(device)

        # Nhúng thông điệp
        with torch.no_grad():
            stego_audio = model(audio_tensor, message_tensor).squeeze().cpu().numpy()

        # Giảm nhiễu với Wavelet Denoising
        stego_audio = wavelet_denoising(stego_audio)
        stego_audio = np.clip(stego_audio, -1.0, 1.0)  # Giới hạn giá trị của âm thanh
        stego_audio = stego_audio / np.max(np.abs(stego_audio))  # Chuẩn hóa lại âm thanh

        return stego_audio, sample_rate

    except Exception as e:
        raise Exception(f"Lỗi khi xử lý: {str(e)}")

def save_audio_file(stego_audio, sample_rate, output_audio="stego_audio.wav"):
    # Lưu file audio đã nhúng thông điệp
    sf.write(output_audio, stego_audio, sample_rate)
    print(f"Đã lưu audio chứa thông điệp tại: {output_audio}")

# GUI sử dụng Tkinter
def gui():
    root = tk.Tk()
    root.title("Audio Steganography")

    def browse_file():
        filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        file_entry.delete(0, tk.END)
        file_entry.insert(0, filepath)

    def browse_model_file():
        filepath = filedialog.askopenfilename(filetypes=[("Model files", "*.pth")])
        model_file_entry.delete(0, tk.END)
        model_file_entry.insert(0, filepath)

    def encode_message():
        input_audio = file_entry.get()
        secret_message = message_entry.get()
        model_path = model_file_entry.get()

        if not secret_message:
            messagebox.showerror("Error", "Vui lòng nhập thông điệp.")
            return
        if not input_audio:
            messagebox.showerror("Error", "Vui lòng chọn tệp âm thanh.")
            return
        if not model_path:
            messagebox.showerror("Error", "Vui lòng chọn tệp mô hình.")
            return

        try:
            stego_audio, sample_rate = encode_audio(input_audio, secret_message, model_path)
            save_audio_file(stego_audio, sample_rate)
            messagebox.showinfo("Success", "Thông điệp đã được nhúng vào âm thanh.")
            root.quit()  # Tự động thoát GUI khi mã hóa thành công
        except Exception as e:
            messagebox.showerror("Error", f"Lỗi: {str(e)}")

    # Giao diện nhập tệp âm thanh
    tk.Label(root, text="Chọn tệp âm thanh (WAV)").pack(pady=5)
    file_entry = tk.Entry(root, width=40)
    file_entry.pack(pady=5)
    browse_button = tk.Button(root, text="Duyệt", command=browse_file)
    browse_button.pack(pady=5)

    # Giao diện nhập tệp mô hình
    tk.Label(root, text="Chọn tệp mô hình Autoencoder (.pth)").pack(pady=5)
    model_file_entry = tk.Entry(root, width=40)
    model_file_entry.pack(pady=5)
    browse_model_button = tk.Button(root, text="Duyệt", command=browse_model_file)
    browse_model_button.pack(pady=5)

    # Giao diện nhập thông điệp
    tk.Label(root, text="Nhập thông điệp bí mật").pack(pady=5)
    message_entry = tk.Entry(root, width=40)
    message_entry.pack(pady=5)

    # Giao diện mã hóa
    encode_button = tk.Button(root, text="Mã hóa và lưu", command=encode_message)
    encode_button.pack(pady=20)

    # Giao diện kết thúc
    root.mainloop()

if __name__ == "__main__":
    gui()
