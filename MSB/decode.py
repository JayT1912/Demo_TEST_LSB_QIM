# decode.py

import wave  # Thư viện xử lý file WAV
import numpy as np  # Thư viện xử lý mảng số
import sys  # Thư viện để đọc tham số dòng lệnh

def decode_msb(encoded_audio_path):
    # Mở file WAV chứa dữ liệu đã giấu
    with wave.open(encoded_audio_path, 'rb') as audio:
        frames = audio.readframes(audio.getnframes())  # Đọc toàn bộ dữ liệu âm thanh

    # Chuyển dữ liệu audio thành mảng numpy kiểu int16
    audio_data = np.frombuffer(frames, dtype=np.int16)

    bits = []  # Danh sách chứa các bit trích xuất

    # Lặp qua từng mẫu âm thanh để lấy MSB
    for sample in audio_data:
        msb = (sample >> 15) & 1  # Dịch 15 bit sang phải để lấy MSB, rồi AND với 1
        bits.append(str(msb))  # Thêm bit vào danh sách

    message = ''  # Biến chứa thông điệp kết quả

    # Gom từng 8 bit thành 1 ký tự
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]  # Lấy 8 bit một lần
        if len(byte) < 8:
            break  # Nếu thiếu bit thì dừng
        char = chr(int(''.join(byte), 2))  # Chuyển bit thành ký tự
        if char == '\x00':  # Nếu gặp NULL character => dừng
            break
        message += char  # Thêm ký tự vào thông điệp

    return message  # Trả về thông điệp giải mã được

if __name__ == "__main__":
    # Kiểm tra số lượng tham số truyền vào
    if len(sys.argv) != 2:
        print("Cách dùng: python decode.py <encoded_audio.wav>")  # Hướng dẫn sử dụng
        sys.exit(1)  # Thoát nếu sai cú pháp

    # Đọc tham số từ dòng lệnh
    encoded_audio = sys.argv[1]  # File WAV đã giấu dữ liệu

    # Gọi hàm decode và in ra thông điệp
    secret_message = decode_msb(encoded_audio)
    print("Thông điệp giải mã được:", secret_message)
