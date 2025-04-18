import numpy as np
import soundfile as sf

# 1. Nhập thông điệp thủ công
def input_message():
    # Nhập thông điệp cần giấu vào âm thanh
    message = input("Nhập thông điệp cần giấu vào âm thanh: ")
    return message

# 2. Chuyển thông điệp thành chuỗi nhị phân
def text_to_binary(text):
    # Thêm chuỗi đánh dấu "START" vào đầu thông điệp
    text = "START" + text + '\0'  # Thêm ký tự kết thúc '\0'
    binary = ''.join(format(ord(c), '08b') for c in text)
    return binary

# 3. Tải tín hiệu âm thanh
def load_audio(filename):
    audio_data, sample_rate = sf.read(filename)
    print(f"Tải âm thanh: {filename}, số mẫu: {len(audio_data)}")
    return audio_data, sample_rate

# 4. Áp dụng QIM (Lượng tử hóa) - Đây là bước tùy chọn
def quantize(samples, step_size):
    print(f"Lượng tử hóa với step_size={step_size}")
    quantized = np.round(samples / step_size) * step_size
    return quantized

# 5. Giấu thông điệp vào LSB
def hide_in_lsb(samples, secret_bits):
    if len(samples.shape) > 1:
        samples = samples[:, 0]  # Lấy kênh đầu tiên nếu âm thanh stereo

    secret_index = 0
    for i in range(len(samples)):
        sample = int(samples[i])  # Chuyển đổi thành số nguyên
        if secret_index < len(secret_bits):
            # Giấu một bit vào LSB
            sample = (sample & 0xFE) | int(secret_bits[secret_index])
            samples[i] = sample
            secret_index += 1
        if secret_index >= len(secret_bits):
            break
    print(f"Đã giấu {secret_index} bits.")
    return samples

# 6. Lưu lại tín hiệu âm thanh đã giấu thông điệp
def save_audio(filename, samples, sample_rate):
    sf.write(filename, np.array(samples), sample_rate)
    print(f"Lưu âm thanh vào: {filename}")

# Quy trình mã hóa (encode)
def encode_message(input_audio_filename, output_audio_filename, step_size=0.1):
    # Nhập thông điệp từ người dùng
    message = input_message()
    
    # Chuyển thông điệp thành chuỗi nhị phân
    binary_message = text_to_binary(message)
    print(f"Thông điệp nhị phân: {binary_message[:50]}...")  # In phần đầu của thông điệp
    
    # Tải âm thanh gốc
    audio_data, sample_rate = load_audio(input_audio_filename)
    
    # Áp dụng QIM (nếu cần) để điều chỉnh mẫu âm thanh
    quantized_data = quantize(audio_data, step_size)
    
    # Giấu thông điệp vào LSB của tín hiệu âm thanh
    hidden_data = hide_in_lsb(quantized_data, binary_message)
    
    # Lưu lại âm thanh đã giấu thông điệp vào tệp mới
    save_audio(output_audio_filename, hidden_data, sample_rate)
    print(f"Thông điệp đã được giấu và lưu vào {output_audio_filename}")

# Sử dụng hàm mã hóa
input_audio_filename = 'input_audio.wav'  # Đường dẫn tới âm thanh gốc
output_audio_filename = 'output_audio.wav'  # Đường dẫn lưu âm thanh đã giấu thông điệp

encode_message(input_audio_filename, output_audio_filename)
