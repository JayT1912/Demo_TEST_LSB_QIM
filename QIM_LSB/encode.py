import numpy as np  # Thư viện numpy để xử lý dữ liệu số (mảng số)
import soundfile as sf  # Thư viện soundfile để đọc và ghi tệp âm thanh (WAV, FLAC,...)
import tkinter as tk  # Thư viện tkinter để xây dựng giao diện người dùng (GUI)
from tkinter import filedialog, messagebox  # Module phụ trợ của tkinter để chọn tệp và hiện thông báo

# Hàm đọc file audio
def load_audio(filename):
    audio_data, sample_rate = sf.read(filename)  # Đọc dữ liệu mẫu âm thanh và tần số lấy mẫu
    return audio_data, sample_rate  # Trả về dữ liệu mẫu và sample_rate

# Hàm lượng tử hóa tín hiệu theo step_size (QIM)
def quantize(samples, step_size):
    return np.round(samples / step_size) * step_size  # Chia mẫu theo khoảng step_size rồi làm tròn

# Hàm chuyển thông điệp thành chuỗi nhị phân
def text_to_binary(text):
    text = "START" + text + '\0'  # Thêm dấu hiệu bắt đầu "START" và kết thúc '\0' vào thông điệp
    return ''.join(format(ord(c), '08b') for c in text)  # Mỗi ký tự chuyển thành 8 bit nhị phân

# Hàm giấu thông điệp vào LSB sau khi đã lượng tử hóa QIM
def hide_in_qim_lsb(samples, secret_bits, step_size):
    if len(samples.shape) > 1:
        samples = samples[:, 0]  # Nếu tín hiệu stereo, chỉ lấy kênh trái (kênh đầu tiên)

    samples = quantize(samples, step_size)  # Bước 1: Lượng tử hóa mẫu theo step_size
    secret_index = 0  # Vị trí của bit thông điệp cần giấu

    for i in range(len(samples)):  # Duyệt từng mẫu tín hiệu
        sample = int(samples[i])  # Chuyển mẫu về số nguyên
        if secret_index < len(secret_bits):  # Nếu còn bit cần giấu
            sample = (sample & 0xFE) | int(secret_bits[secret_index])  # Gán bit vào LSB (bit thấp nhất)
            samples[i] = sample  # Cập nhật mẫu
            secret_index += 1  # Chuyển sang bit tiếp theo
        if secret_index >= len(secret_bits):  # Nếu đã giấu hết thông điệp
            break

    return samples  # Trả về mẫu tín hiệu đã giấu thông điệp

# Hàm lưu file âm thanh
def save_audio(filename, samples, sample_rate):
    sf.write(filename, np.array(samples), sample_rate)  # Ghi mẫu tín hiệu vào file WAV mới

# Hàm xử lý encode toàn bộ quy trình
def encode_message():
    if not audio_filename:  # Nếu chưa chọn file
        messagebox.showerror("Lỗi", "Vui lòng chọn file âm thanh!")  # Hiển thị lỗi
        return

    message = message_entry.get()  # Lấy thông điệp người dùng nhập
    if not message:  # Nếu thông điệp rỗng
        messagebox.showerror("Lỗi", "Vui lòng nhập thông điệp cần giấu!")  # Hiển thị lỗi
        return

    try:
        step_size = float(step_entry.get())  # Lấy giá trị step_size từ ô nhập
        if step_size <= 0:  # Kiểm tra step_size hợp lệ
            raise ValueError
    except ValueError:
        messagebox.showerror("Lỗi", "Vui lòng nhập Step Size hợp lệ (> 0)!")  # Hiển thị lỗi nếu step_size không hợp lệ
        return

    binary_message = text_to_binary(message)  # Chuyển thông điệp thành chuỗi bit
    audio_data, sample_rate = load_audio(audio_filename)  # Đọc file âm thanh
    hidden_data = hide_in_qim_lsb(audio_data, binary_message, step_size)  # Giấu thông điệp vào tín hiệu

    # Mở hộp thoại lưu file kết quả
    output_filename = filedialog.asksaveasfilename(
        defaultextension=".wav",  # Mặc định lưu file dạng WAV
        filetypes=[("WAV files", "*.wav")],  # Chỉ cho phép lưu WAV
        title="Lưu file âm thanh mới"
    )

    if output_filename:  # Nếu người dùng đã chọn nơi lưu
        save_audio(output_filename, hidden_data, sample_rate)  # Ghi file mới
        messagebox.showinfo("Thành công", f"Đã mã hóa và lưu vào:\n{output_filename}")  # Thông báo lưu thành công

# Hàm chọn file âm thanh nguồn
def browse_audio():
    global audio_filename  # Dùng biến toàn cục để lưu tên file
    audio_filename = filedialog.askopenfilename(
        filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.wma *.aiff *.alac"),
                   ("All files", "*.*")]
    )
    if audio_filename:  # Nếu chọn file thành công
        file_label.config(text=f"Đã chọn: {audio_filename}")  # Hiển thị tên file đã chọn

# ======================== Giao diện Tkinter (Cải tiến) ========================

root = tk.Tk()  # Tạo cửa sổ chính
root.title("Giấu thông điệp (QIM + LSB)")  # Tiêu đề cửa sổ
root.geometry("600x500")  # Kích thước cửa sổ
root.config(bg="#e8f0fe")  # Màu nền sáng dịu

audio_filename = None  # Biến lưu tên file audio

# Khung chính
main_frame = tk.Frame(root, bg="#ffffff", relief="groove", bd=2)
main_frame.pack(padx=20, pady=20, fill="both", expand=True)

# Tiêu đề
title_label = tk.Label(main_frame, text="Giấu Thông Điệp (QIM + LSB)", font=("Helvetica", 16, "bold"), bg="#ffffff", fg="#333")
title_label.pack(pady=10)

# Nút chọn file
browse_button = tk.Button(main_frame, text="Chọn File Âm Thanh", command=browse_audio, font=("Helvetica", 12), bg="#4CAF50", fg="white", relief="flat", padx=20, pady=10)
browse_button.pack(pady=10)

file_label = tk.Label(main_frame, text="Chưa chọn file", font=("Helvetica", 10), fg="gray", bg="#ffffff")
file_label.pack()

# Nhãn và ô nhập thông điệp
message_label = tk.Label(main_frame, text="Thông điệp cần giấu:", font=("Helvetica", 12), bg="#ffffff")
message_label.pack(pady=10)

message_entry = tk.Entry(main_frame, font=("Helvetica", 12), width=45, relief="solid", bd=2, fg="#333")
message_entry.pack(pady=5)

# Nhãn và ô nhập step size
step_label = tk.Label(main_frame, text="Step Size (ví dụ 0.1):", font=("Helvetica", 12), bg="#ffffff")
step_label.pack(pady=10)

step_entry = tk.Entry(main_frame, font=("Helvetica", 12), width=10, relief="solid", bd=2, fg="#333")
step_entry.insert(0, "0.1")  # Giá trị mặc định là 0.1
step_entry.pack(pady=5)

# Nút mã hóa
encode_button = tk.Button(main_frame, text="Mã hóa và Lưu File", command=encode_message, font=("Helvetica", 12), bg="#008CBA", fg="white", relief="flat", padx=20, pady=10)
encode_button.pack(pady=20)

# Ghi chú
footer_label = tk.Label(main_frame, text="© 2025 - Công cụ giải mã thông điệp", font=("Helvetica", 9), bg="#ffffff", fg="#888")
footer_label.pack(side="bottom", pady=10)

# Khởi chạy vòng lặp giao diện
root.mainloop()