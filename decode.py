import numpy as np
import soundfile as sf

# Hàm giải mã nhị phân từ âm thanh
def decode_binary_from_audio(audio_data, num_bits):
    # Giải mã các bit nhúng vào âm thanh
    binary_message = ""
    for i in range(num_bits):
        binary_message += str(audio_data[i] & 1)  # Lấy bit cuối cùng của mỗi mẫu âm thanh

    return binary_message

# Hàm chuyển đổi nhị phân thành văn bản
def binary_to_text(binary_data):
    # Chia chuỗi nhị phân thành các đoạn 8 bit (1 byte)
    byte_chunks = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]
    
    # Chuyển đổi từng byte nhị phân thành ký tự
    text = "".join(chr(int(byte, 2)) for byte in byte_chunks)
    
    return text

# ==== CHẠY ==== 

# Đọc tệp âm thanh có thông điệp đã nhúng
audio_data, sr = sf.read('output_audio.wav', dtype='int16')

# Giả sử bạn biết trước số lượng bit cần giải mã (ví dụ 1000 bit)
num_bits_to_decode = 1000

# Giải mã nhị phân từ âm thanh
binary_data = decode_binary_from_audio(audio_data, num_bits_to_decode)

# Chuyển nhị phân thành văn bản
decoded_message = binary_to_text(binary_data)
print("Thông điệp giải mã: ", decoded_message)
