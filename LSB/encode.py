import numpy as np
import soundfile as sf

# Hàm chuyển đổi thông điệp sang nhị phân
def text_to_binary(text):
    return ''.join(format(ord(char), '08b') for char in text)

# Hàm nhúng thông điệp vào âm thanh
def embed_binary_in_audio(audio_data, binary_message):
    sr = 44100  # Giả sử tần số mẫu mặc định là 44.1 kHz
    audio = np.array(audio_data, dtype='int16')
    
    # Nếu âm thanh là stereo, chỉ lấy kênh đầu tiên (mono)
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # mono
    
    # Kiểm tra độ dài thông điệp
    if len(binary_message) > len(audio):
        raise ValueError("Thông điệp quá dài.")
    
    # Nhúng thông điệp vào tín hiệu âm thanh
    modified_audio = np.copy(audio)
    for i in range(len(binary_message)):
        modified_audio[i] = (audio[i] & ~1) | int(binary_message[i])

    # Lưu tệp âm thanh đã giấu tin
    sf.write('output_audio.wav', modified_audio, sr)
    print("Đã tạo output_audio.wav với thông điệp nhúng.")

# ==== CHẠY ==== 

# Nhập thông điệp từ người dùng
message = input("Nhập thông điệp bạn muốn nhúng vào âm thanh: ") + '###END###'  # Thêm dấu kết thúc thông điệp

# Chuyển đổi thông điệp thành chuỗi nhị phân
binary_data = text_to_binary(message)

# Ghi ra file nhị phân
with open('tepNhiPhan.txt', 'w', encoding='utf-8') as f:
    f.write(binary_data)
print("Đã ghi chuỗi nhị phân vào tepNhiPhan.txt")

# Đọc tệp âm thanh input_audio.wav
audio_data, sr = sf.read('input_audio.wav', dtype='int16')

# Nhúng thông điệp vào tệp âm thanh
embed_binary_in_audio(audio_data, binary_data)
