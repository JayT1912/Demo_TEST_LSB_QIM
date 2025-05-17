import numpy as np  # Thư viện xử lý số học
import soundfile as sf  # Thư viện đọc file âm thanh
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ
from matplotlib.backends.backend_pdf import PdfPages  # Module lưu nhiều biểu đồ vào file PDF
import tkinter as tk  # Thư viện tạo giao diện GUI
from tkinter import filedialog, messagebox  # Các tiện ích chọn file, hiển thị thông báo

# Hàm tính SNR
def calculate_snr(original, modified):
    noise = original - modified  # Hiệu giữa tín hiệu gốc và encode
    signal_power = np.mean(original ** 2)  # Công suất tín hiệu
    noise_power = np.mean(noise ** 2)  # Công suất nhiễu
    if noise_power == 0:
        return np.inf  # Trường hợp tín hiệu giống hệt nhau
    snr = 10 * np.log10(signal_power / noise_power)  # Công thức tính SNR
    return snr

# Hàm đọc file và chuẩn hóa tín hiệu
def load_audio_files(original_file, modified_file):
    original_audio, sr1 = sf.read(original_file)  # Đọc file gốc
    modified_audio, sr2 = sf.read(modified_file)  # Đọc file encode

    if sr1 != sr2:  # Kiểm tra tần số mẫu giống nhau
        raise ValueError("Hai file có sample rate khác nhau!")

    if len(original_audio.shape) > 1:
        original_audio = original_audio[:, 0]  # Lấy 1 kênh nếu stereo
    if len(modified_audio.shape) > 1:
        modified_audio = modified_audio[:, 0]

    length = min(len(original_audio), len(modified_audio))  # Đồng bộ độ dài
    time = np.linspace(0, length / sr1, num=length)  # Tạo trục thời gian

    return original_audio[:length], modified_audio[:length], time, sr1

# Hàm so sánh và vẽ waveform
def compare_audio():
    try:
        original_audio, modified_audio, time, sr = load_audio_files(original_audio_filename, modified_audio_filename)
    except Exception as e:
        messagebox.showerror("Lỗi", str(e))
        return

    snr_value = calculate_snr(original_audio, modified_audio)

    # Hiển thị thông báo dựa trên SNR
    if snr_value > 100:
        messagebox.showinfo("Thông báo", f"Hai file gần như giống nhau (SNR = {snr_value:.2f} dB).")
    elif snr_value < 10:
        messagebox.showwarning("Cảnh báo", f"Hai file khác biệt quá lớn (SNR = {snr_value:.2f} dB).")
    else:
        messagebox.showinfo("Kết quả", f"So sánh thành công!\nSNR = {snr_value:.2f} dB.")

    # Vẽ waveform
    plt.figure(figsize=(12, 6))
    plt.plot(time, original_audio, label='Âm thanh gốc', alpha=0.7)
    plt.plot(time, modified_audio, label='Âm thanh (QIM+LSB)', alpha=0.7)
    plt.title(f"So sánh tín hiệu âm thanh\nSNR = {snr_value:.2f} dB")
    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Biên độ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Hàm xuất báo cáo PDF (chỉ lưu waveform)
def export_pdf():
    try:
        original_audio, modified_audio, time, sr = load_audio_files(original_audio_filename, modified_audio_filename)
    except Exception as e:
        messagebox.showerror("Lỗi", str(e))
        return

    snr_value = calculate_snr(original_audio, modified_audio)

    save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")],
                                             title="Chọn nơi lưu báo cáo PDF")
    if not save_path:
        return

    pdf = PdfPages(save_path)  # Tạo file PDF

    # Vẽ và lưu waveform
    plt.figure(figsize=(12, 6))
    plt.plot(time, original_audio, label='Âm thanh gốc', alpha=0.7)
    plt.plot(time, modified_audio, label='Âm thanh (QIM+LSB)', alpha=0.7)
    plt.title(f"So sánh tín hiệu âm thanh\nSNR = {snr_value:.2f} dB")
    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Biên độ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()  # Lưu hình vào file PDF
    plt.close()

    pdf.close()  # Đóng file PDF

    messagebox.showinfo("Hoàn thành", f"Báo cáo PDF đã lưu vào:\n{save_path}")

# Hàm chọn file âm thanh gốc
def browse_original():
    global original_audio_filename
    original_audio_filename = filedialog.askopenfilename(
        filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.wma *.aiff *.alac"),
                   ("All files", "*.*")]
    )
    if original_audio_filename:
        original_file_label.config(text=f"File gốc: {original_audio_filename}")

# Hàm chọn file encode
def browse_modified():
    global modified_audio_filename
    modified_audio_filename = filedialog.askopenfilename(
        filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.wma *.aiff *.alac"),
                   ("All files", "*.*")]
    )
    if modified_audio_filename:
        modified_file_label.config(text=f"File giấu tin: {modified_audio_filename}")

# ======= Giao diện Tkinter =======

root = tk.Tk()
root.title("So sánh Âm thanh và Xuất báo cáo PDF (QIM+LSB)")
root.geometry("650x450")

original_audio_filename = None
modified_audio_filename = None

# Nút chọn file gốc
browse_original_button = tk.Button(root, text="Chọn file âm thanh gốc", command=browse_original,
                                   font=("Arial", 12), bg="#4CAF50", fg="white", width=30)
browse_original_button.pack(pady=15)

original_file_label = tk.Label(root, text="Chưa chọn file gốc", font=("Arial", 10), fg="gray")
original_file_label.pack()

# Nút chọn file encode
browse_modified_button = tk.Button(root, text="Chọn file âm thanh đã giấu tin", command=browse_modified,
                                   font=("Arial", 12), bg="#2196F3", fg="white", width=30)
browse_modified_button.pack(pady=15)

modified_file_label = tk.Label(root, text="Chưa chọn file  đã giấu tin", font=("Arial", 10), fg="gray")
modified_file_label.pack()

# Nút so sánh waveform
compare_button = tk.Button(root, text="So sánh âm thanh", command=compare_audio,
                           font=("Arial", 14), bg="#f44336", fg="white", width=30)
compare_button.pack(pady=20)

# Nút export báo cáo PDF
export_button = tk.Button(root, text="Xuất báo cáo PDF", command=export_pdf,
                          font=("Arial", 14), bg="#9C27B0", fg="white", width=30)
export_button.pack(pady=10)

root.mainloop()
