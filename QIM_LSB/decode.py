import numpy as np
import soundfile as sf
import os

# 1. Chuyển chuỗi nhị phân thành văn bản
def binary_to_text(binary_str):
    text = ''
    for i in range(0, len(binary_str), 8):
        char = chr(int(binary_str[i:i+8], 2))
        if char == '\0':  # Dừng khi gặp ký tự kết thúc
            break
        text += char
    return text

# 2. Trích xuất thông điệp từ LSB
def extract_from_lsb(samples, num_bits):
    # Nếu âm thanh là stereo, chỉ lấy kênh đầu tiên
    if len(samples.shape) > 1:
        samples = samples[:, 0]  # Lấy kênh trái (kênh đầu tiên)

    extracted_bits = []
    for i in range(len(samples)):
        sample = int(np.round(samples[i]))  # Làm tròn và chuyển đổi thành số nguyên
        extracted_bits.append(sample & 1)  # Lấy bit LSB
        if len(extracted_bits) >= num_bits:
            break
    return extracted_bits

# 3. Tải tín hiệu âm thanh
def load_audio(filename):
    audio_data, sample_rate = sf.read(filename)
    print(f"Tải âm thanh: {filename}, số mẫu: {len(audio_data)}")
    return audio_data, sample_rate

# Quy trình giải mã (decode)
def decode_message(input_audio_filename):
    # Tải âm thanh
    audio_data, sample_rate = load_audio(input_audio_filename)

    # Trích xuất các bit đã giấu từ LSB
    extracted_bits = extract_from_lsb(audio_data, len(audio_data) * 8)  # Trích xuất toàn bộ bit
    
    # Chuyển các bit đã trích xuất thành chuỗi nhị phân
    extracted_binary = ''.join(str(bit) for bit in extracted_bits)
    
    # Chuyển chuỗi nhị phân thành văn bản
    decoded_message = binary_to_text(extracted_binary)
    
    # Kiểm tra chuỗi đánh dấu
    if decoded_message.startswith("START"):
        # Loại bỏ chuỗi "START" trước khi hiển thị thông điệp
        decoded_message = decoded_message[5:]  # Bỏ 5 ký tự đầu tiên ("START")
        print(f"Thông điệp đã giải mã: {decoded_message}")
    else:
        print("Đây là đoạn âm thanh gốc, không chứa thông điệp.")
    
    return decoded_message

# Sử dụng hàm giải mã
input_audio_filename = input("Nhập tên tệp âm thanh cần giải mã: ")  # Nhập tên tệp từ người dùng

# Kiểm tra tệp có tồn tại hay không
if not os.path.exists(input_audio_filename):
    print("Không tìm thấy tệp âm thanh. Vui lòng kiểm tra lại tên tệp.")
else:
    decoded_message = decode_message(input_audio_filename)