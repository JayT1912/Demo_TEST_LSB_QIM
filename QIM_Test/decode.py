import numpy as np
import wave
import struct

# Hàm lượng hóa (quantization)
def quantize(signal, step_size):
    return np.round(signal / step_size) * step_size

# Hàm giải mã (Decoding)
def decode_message(received_signal, original_signal, step_size, message_length):
    decoded_bits = []
    
    # Lượng hóa tín hiệu gốc và tín hiệu nhận được
    quantized_original = quantize(original_signal, step_size)
    quantized_received = quantize(received_signal, step_size)
    
    # Giải mã thông điệp từ tín hiệu nhận được
    bit_index = 0
    for i in range(len(received_signal)):
        if bit_index < message_length * 8:  # Mỗi ký tự có 8 bit
            if quantized_received[i] > quantized_original[i]:
                decoded_bits.append(1)  # Nếu tín hiệu đã thay đổi, bit = 1
            else:
                decoded_bits.append(0)  # Nếu tín hiệu không thay đổi, bit = 0
            bit_index += 1
    
    # Chuyển các bit thành chuỗi ký tự
    decoded_message = ''.join(str(bit) for bit in decoded_bits)
    decoded_chars = [chr(int(decoded_message[i:i+8], 2)) for i in range(0, len(decoded_message), 8)]
    return ''.join(decoded_chars)

# Đọc tín hiệu âm thanh từ file WAV
def read_audio(filename):
    wav_file = wave.open(filename, 'rb')
    n_channels = wav_file.getnchannels()
    sampwidth = wav_file.getsampwidth()
    framerate = wav_file.getframerate()
    n_samples = wav_file.getnframes()

    if sampwidth != 2:
        raise ValueError("File WAV phải sử dụng độ sâu mẫu 16-bit (2 bytes).")

    raw_data = wav_file.readframes(n_samples)
    wav_file.close()

    # Chuyển đổi dữ liệu âm thanh thành mảng numpy
    audio_signal = np.array(struct.unpack('<' + str(n_samples * n_channels) + 'h', raw_data))

    # Nếu file là stereo, chuyển đổi sang mono bằng cách lấy trung bình 2 kênh
    if n_channels == 2:
        audio_signal = audio_signal.reshape(-1, 2).mean(axis=1).astype(np.int16)

    return audio_signal, len(audio_signal)

# Yêu cầu người dùng nhập thông tin giải mã
modified_audio_file = input("Nhập tên file âm thanh đã giấu thông điệp: ")  # Tín hiệu âm thanh đã giấu thông điệp
audio_file = 'input_audio.wav'  # Tín hiệu âm thanh gốc

# Đọc tín hiệu âm thanh
audio_signal, _ = read_audio(audio_file)
modified_signal, _ = read_audio(modified_audio_file)

# Kích thước bước lượng hóa
step_size = 100

# Độ dài thông điệp (giả sử bạn biết trước, ví dụ: 5 ký tự)
message_length = 1000  # Thay giá trị này bằng độ dài thực tế của thông điệp

# Giải mã lại thông điệp từ tín hiệu đã nhận
decoded_message = decode_message(modified_signal, audio_signal, step_size, message_length)

print(f"Thông điệp đã giải mã: {decoded_message}")