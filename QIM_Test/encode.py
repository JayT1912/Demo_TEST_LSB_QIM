import numpy as np
import wave
import struct

# Hàm lượng hóa (quantization)
def quantize(signal, step_size):
    return np.round(signal / step_size) * step_size

# Hàm giấu thông điệp (Embedding)
def embed_message(audio_signal, secret_message, step_size):
    # Chuyển thông điệp thành chuỗi nhị phân
    message_bits = ''.join(format(ord(char), '08b') for char in secret_message)
    
    # Lượng hóa tín hiệu âm thanh
    quantized_signal = quantize(audio_signal, step_size)
    
    # Giấu thông điệp vào tín hiệu
    bit_index = 0
    for i in range(len(audio_signal)):
        if bit_index < len(message_bits):
            # Giấu một bit thông điệp vào mỗi giá trị tín hiệu
            if message_bits[bit_index] == '1':
                quantized_signal[i] += step_size  # Tăng giá trị tín hiệu
            bit_index += 1
    
    return quantized_signal

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

# Ghi tín hiệu âm thanh vào file WAV
def write_audio(filename, audio_signal):
    wav_file = wave.open(filename, 'wb')
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 2 bytes per sample
    wav_file.setframerate(44100)  # 44.1 kHz sample rate

    # Đảm bảo audio_signal là mảng kiểu int16
    audio_signal = np.asarray(audio_signal, dtype=np.int16)

    wav_file.writeframes(struct.pack('<' + str(len(audio_signal)) + 'h', *audio_signal))
    wav_file.close()

# Yêu cầu người dùng nhập thông điệp
secret_message = input("Nhập thông điệp bạn muốn giấu vào âm thanh: ")

# Đọc tín hiệu âm thanh gốc
audio_file = 'input_audio.wav'  # Tín hiệu âm thanh gốc
modified_audio_file = 'output_audio.wav'  # Tín hiệu âm thanh đã giấu thông điệp
audio_signal, n_samples = read_audio(audio_file)

# Kích thước bước lượng hóa
step_size = 100

# Giấu thông điệp vào tín hiệu âm thanh
modified_signal = embed_message(audio_signal, secret_message, step_size)

# Ghi tín hiệu âm thanh đã giấu thông điệp vào file mới
write_audio(modified_audio_file, modified_signal)

print(f"Thông điệp đã được giấu vào file {modified_audio_file}.")
