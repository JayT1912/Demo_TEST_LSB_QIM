# encode.py

import wave  # Thư viện xử lý file WAV
import numpy as np  # Thư viện xử lý mảng số
import sys  # Thư viện để đọc tham số dòng lệnh

def encode_msb(audio_path, output_path, secret_message):
    # Mở file audio WAV ở chế độ đọc
    with wave.open(audio_path, 'rb') as audio:
        params = audio.getparams()  # Lấy các tham số (số kênh, sample width, framerate, etc.)
        frames = audio.readframes(params.nframes)  # Đọc toàn bộ dữ liệu âm thanh

    # Chuyển dữ liệu âm thanh thành mảng numpy kiểu int16 (âm thanh PCM 16 bit)
    audio_data = np.frombuffer(frames, dtype=np.int16)

    # Chuyển thông điệp thành chuỗi bit (mỗi ký tự thành 8 bit nhị phân)
    secret_bits = ''.join(f'{ord(c):08b}' for c in secret_message)
    secret_bits += '00000000'  # Thêm ký tự kết thúc: NULL (8 bit 0)

    # Kiểm tra nếu thông điệp quá dài so với số mẫu âm thanh
    if len(secret_bits) > len(audio_data):
        raise ValueError("Dữ liệu cần giấu quá lớn so với file audio!")

    # Tạo một bản sao dữ liệu âm thanh để chỉnh sửa
    modified_audio = np.copy(audio_data)

    # Lặp qua từng bit của thông điệp và gán vào MSB của các mẫu âm thanh
    for i, bit in enumerate(secret_bits):
        if bit == '0':
            modified_audio[i] = modified_audio[i] & 0x7FFF  # Đặt MSB = 0 bằng cách AND với 0111 1111 1111 1111
        else:
            modified_audio[i] = modified_audio[i] | 0x8000  # Đặt MSB = 1 bằng cách OR với 1000 0000 0000 0000

    # Ghi dữ liệu âm thanh đã chỉnh sửa ra file WAV mới
    with wave.open(output_path, 'wb') as output_audio:
        output_audio.setparams(params)  # Gán lại tham số âm thanh ban đầu
        output_audio.writeframes(modified_audio.tobytes())  # Ghi dữ liệu ra file

    print(f"Đã encode thành công! File lưu tại: {output_path}")  # Thông báo hoàn thành

if __name__ == "__main__":
    # Kiểm tra số lượng tham số truyền vào chương trình
    if len(sys.argv) != 4:
        print("Cách dùng: python encode.py <input_audio.wav> <output_audio.wav> <secret_message>")  # Hướng dẫn sử dụng
        sys.exit(1)  # Thoát nếu sai cú pháp

    # Đọc tham số từ dòng lệnh
    input_audio = sys.argv[1]  # File WAV đầu vào
    output_audio = sys.argv[2]  # File WAV đầu ra
    secret_message = sys.argv[3]  # Thông điệp bí mật cần nhúng

    # Gọi hàm encode
    encode_msb(input_audio, output_audio, secret_message)
