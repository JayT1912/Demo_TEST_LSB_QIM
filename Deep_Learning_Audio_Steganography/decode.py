import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import os
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import tkinter as tk  # Thêm dòng này để import tkinter
from tkinter import filedialog, messagebox
import threading
import queue
from scipy.signal import butter, filtfilt, savgol_filter

# Định nghĩa mô hình Autoencoder với PyTorch
class StegoAutoencoder(nn.Module):
    def __init__(self):
        super(StegoAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, audio, message):
        x = torch.cat((audio, message), dim=1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Hàm giải mã thông điệp từ âm thanh
def extract_message_from_audio(audio, model, device):
    """Giải mã thông điệp từ âm thanh đã nhúng thông qua mô hình"""
    
    # Tạo tensor message ban đầu với giá trị 0
    message_tensor = torch.zeros((1, 1, audio.shape[2]), dtype=torch.float32).to(device)

    with torch.no_grad():
        # Dự đoán tín hiệu giải mã từ mô hình
        decoded_audio = model(audio, message_tensor)
    
    # Chuyển tín hiệu giải mã về numpy array
    decoded_audio = decoded_audio.squeeze().cpu().numpy()

    # Làm mượt tín hiệu để giảm nhiễu
    decoded_audio = smooth_signal(decoded_audio)

    # Thử nghiệm với ngưỡng linh động
    binary_message = ''.join(['1' if abs(sample) > 0.05 else '0' for sample in decoded_audio])

    # Đảm bảo chuỗi nhị phân có độ dài là bội số của 8
    if len(binary_message) % 8 != 0:
        binary_message = binary_message[:-(len(binary_message) % 8)]  # Cắt bớt phần dư

    return binary_to_string(binary_message)


# Hàm chuyển nhị phân thành văn bản
def binary_to_string(binary):
    """Chuyển chuỗi nhị phân thành văn bản"""
    try:
        chars = [binary[i:i+8] for i in range(0, len(binary), 8)]
        message = ''.join(chr(int(char, 2)) for char in chars if len(char) == 8)
        return message
    except Exception as e:
        print(f"Lỗi khi chuyển đổi nhị phân thành chuỗi: {str(e)}")
        return ""


def low_pass_filter(audio, cutoff=3000, fs=16000, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_audio = filtfilt(b, a, audio)
    return filtered_audio


def smooth_signal(audio, window_length=11, polyorder=2):
    return savgol_filter(audio, window_length=window_length, polyorder=polyorder)


# Hàm giải mã từ tệp âm thanh đã nhúng thông điệp
def decode_message(input_audio, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StegoAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Đọc tệp âm thanh đã nhúng thông điệp
    audio, sr = sf.read(input_audio)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    secret_message = extract_message_from_audio(audio_tensor, model, device)
    return secret_message


# Giao diện giải mã thông điệp từ âm thanh
def gui_decode():
    def browse_audio_file():
        file_selected = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_selected:
            audio_file_entry.delete(0, tk.END)
            audio_file_entry.insert(0, file_selected)

    def browse_model_file():
        file_selected = filedialog.askopenfilename(filetypes=[("Model files", "*.pth")])
        if file_selected:
            model_file_entry.delete(0, tk.END)
            model_file_entry.insert(0, file_selected)

    def decode_message_gui():
        input_audio = audio_file_entry.get()
        model_path = model_file_entry.get()

        if not input_audio:
            messagebox.showerror("Error", "Vui lòng chọn tệp âm thanh.")
            return
        if not model_path:
            messagebox.showerror("Error", "Vui lòng chọn tệp mô hình.")
            return

        if not os.path.exists(input_audio):
            messagebox.showerror("Error", f"File âm thanh '{input_audio}' không tồn tại.")
            return
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"File mô hình '{model_path}' không tồn tại.")
            return

        try:
            secret_message = decode_message(input_audio, model_path)
            if secret_message:
                messagebox.showinfo("Decoded Message", f"Thông điệp bí mật: {secret_message}")
            else:
                messagebox.showerror("Error", "Không thể giải mã thông điệp.")
        except Exception as e:
            messagebox.showerror("Error", f"Lỗi khi xử lý: {str(e)}")

    root = tk.Tk()
    root.title("Giải mã thông điệp từ âm thanh")

    tk.Label(root, text="Chọn tệp âm thanh đã nhúng thông điệp (WAV)").pack(pady=5)
    audio_file_entry = tk.Entry(root, width=40)
    audio_file_entry.pack(pady=5)
    browse_audio_button = tk.Button(root, text="Duyệt", command=browse_audio_file)
    browse_audio_button.pack(pady=5)

    tk.Label(root, text="Chọn tệp mô hình Autoencoder (.pth)").pack(pady=5)
    model_file_entry = tk.Entry(root, width=40)
    model_file_entry.pack(pady=5)
    browse_model_button = tk.Button(root, text="Duyệt", command=browse_model_file)
    browse_model_button.pack(pady=5)

    decode_button = tk.Button(root, text="Giải mã thông điệp", command=decode_message_gui)
    decode_button.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    gui_decode()
