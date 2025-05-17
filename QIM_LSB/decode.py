import numpy as np  # Thư viện numpy để xử lý dữ liệu số (mảng dữ liệu âm thanh)
import soundfile as sf  # Thư viện soundfile để đọc và ghi các tệp âm thanh (WAV)
import os  # Thư viện os để làm việc với hệ thống tệp
import tkinter as tk  # Thư viện tkinter để tạo giao diện người dùng (UI)
from tkinter import filedialog, messagebox  # filedialog để chọn tệp, messagebox để hiển thị thông báo

# 1. Chuyển chuỗi nhị phân thành văn bản
def binary_to_text(binary_str):
    text = ''  # Khởi tạo chuỗi văn bản rỗng
    for i in range(0, len(binary_str), 8):  # Lặp qua mỗi 8 bit (1 byte)
        char = chr(int(binary_str[i:i+8], 2))  # Chuyển nhóm 8 bit thành ký tự
        if char == '\0':  # Dừng khi gặp ký tự kết thúc chuỗi
            break
        text += char  # Thêm ký tự vào chuỗi văn bản
    return text  # Trả về văn bản đã giải mã

# 2. Trích xuất thông điệp từ LSB (Least Significant Bit - bit thấp nhất)
def extract_from_lsb(samples, num_bits):
    # Nếu âm thanh là stereo, chỉ lấy kênh đầu tiên (kênh trái)
    if len(samples.shape) > 1:
        samples = samples[:, 0]  # Chỉ lấy kênh trái (kênh đầu tiên)

    extracted_bits = []  # Khởi tạo danh sách lưu trữ các bit đã trích xuất
    for i in range(len(samples)):  # Duyệt qua tất cả các mẫu âm thanh
        sample = int(np.round(samples[i]))  # Làm tròn và chuyển đổi mẫu âm thanh thành số nguyên
        extracted_bits.append(sample & 1)  # Lấy bit thấp nhất (LSB) từ mẫu âm thanh
        if len(extracted_bits) >= num_bits:  # Dừng khi đã lấy đủ số bit cần thiết
            break
    return extracted_bits  # Trả về danh sách các bit đã trích xuất

# 3. Tải tín hiệu âm thanh từ tệp WAV
def load_audio(filename):
    audio_data, sample_rate = sf.read(filename)  # Đọc dữ liệu âm thanh từ tệp
    return audio_data, sample_rate  # Trả về dữ liệu âm thanh và tỷ lệ mẫu

# Quy trình giải mã (decode) thông điệp đã giấu trong âm thanh
def decode_message(input_audio_filename):
    # Tải âm thanh từ tệp
    audio_data, sample_rate = load_audio(input_audio_filename)

    # Trích xuất các bit đã giấu từ LSB (toàn bộ bit)
    extracted_bits = extract_from_lsb(audio_data, len(audio_data) * 8)
    
    # Chuyển các bit đã trích xuất thành chuỗi nhị phân
    extracted_binary = ''.join(str(bit) for bit in extracted_bits)
    
    # Chuyển chuỗi nhị phân thành văn bản
    decoded_message = binary_to_text(extracted_binary)
    
    # Kiểm tra chuỗi đánh dấu "START" để xác nhận đây là thông điệp đã giấu
    if decoded_message.startswith("START"):
        # Loại bỏ chuỗi "START" khỏi thông điệp trước khi hiển thị
        decoded_message = decoded_message[5:]  # Bỏ 5 ký tự đầu tiên ("START")
        return decoded_message  # Trả về thông điệp đã giải mã
    else:
        return "Đây là đoạn âm thanh gốc, không chứa thông điệp."  # Nếu không có thông điệp

# Giao diện người dùng (UI) sử dụng tkinter
def browse_file():
    global audio_filename  # Biến toàn cục để lưu tên tệp âm thanh
    audio_filename = filedialog.askopenfilename(filetypes=[("Tệp WAV", "*.wav")])  # Mở hộp thoại chọn tệp WAV
    if audio_filename:  # Nếu chọn tệp thành công
        file_label.config(text=f"Đã chọn tệp: {audio_filename}")  # Hiển thị tên tệp đã chọn

def decode_audio():
    if not audio_filename:  # Kiểm tra nếu không chọn tệp
        messagebox.showerror("Lỗi", "Vui lòng chọn tệp âm thanh để giải mã!")  # Thông báo lỗi nếu không chọn tệp
        return

    decoded_message = decode_message(audio_filename)  # Giải mã thông điệp từ tệp âm thanh đã chọn
    result_text.delete(1.0, tk.END)  # Xóa nội dung cũ trong Text widget
    result_text.insert(tk.END, decoded_message)  # Hiển thị kết quả giải mã vào ô văn bản

# ======================== Giao diện Tkinter (Cải tiến) ========================

root = tk.Tk()  # Tạo cửa sổ chính
root.title("Giải mã Thông điệp từ Âm thanh")  # Tiêu đề cửa sổ
root.geometry("600x500")  # Kích thước cửa sổ
root.config(bg="#e8f0fe")  # Màu nền sáng dịu

audio_filename = None  # Biến lưu tên file audio

# Khung chính
main_frame = tk.Frame(root, bg="#ffffff", relief="groove", bd=2)
main_frame.pack(padx=20, pady=20, fill="both", expand=True)

# Tiêu đề
title_label = tk.Label(main_frame, text="Giải mã Thông điệp từ Âm thanh", font=("Helvetica", 16, "bold"), bg="#ffffff", fg="#333")
title_label.pack(pady=10)

# Nút chọn tệp âm thanh
file_button = tk.Button(main_frame, text="Chọn Tệp Âm thanh", command=browse_file, font=("Helvetica", 12), bg="#4CAF50", fg="white", relief="flat", padx=20, pady=10)
file_button.pack(pady=10)

# Nhãn hiển thị tên tệp đã chọn
file_label = tk.Label(main_frame, text="Chưa chọn tệp", font=("Helvetica", 10), fg="gray", bg="#ffffff")
file_label.pack(pady=5)

# Nút giải mã thông điệp
decode_button = tk.Button(main_frame, text="Giải mã Thông điệp", command=decode_audio, font=("Helvetica", 12), bg="#008CBA", fg="white", relief="flat", padx=20, pady=10)
decode_button.pack(pady=20)

# Text widget để hiển thị thông điệp đã giải mã
result_text = tk.Text(main_frame, height=10, width=60, font=("Helvetica", 12), wrap=tk.WORD, relief="solid", bd=2, fg="#333")
result_text.pack(pady=10)
# Ghi chú
footer_label = tk.Label(main_frame, text="© 2025 - Công cụ giải mã thông điệp", font=("Helvetica", 9), bg="#ffffff", fg="#888")
footer_label.pack(side="bottom", pady=10)

# Chạy vòng lặp sự kiện của tkinter
root.mainloop()