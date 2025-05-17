import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import soundfile as sf

# Định nghĩa mô hình Autoencoder
class AudioAutoencoder(nn.Module):
    def __init__(self):
        super(AudioAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(13, 64),  # Đầu vào là 13 đặc trưng MFCC
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 13)  # Đầu ra là 13 đặc trưng MFCC
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Hàm giấu thông điệp vào âm thanh
def embed_message(audio_file, message, model, output_file):
    # Sử dụng librosa để đọc file âm thanh
    y, sr = librosa.load(audio_file, sr=None)  # Đọc file âm thanh

    # Bổ sung zero-padding nếu tín hiệu quá ngắn
    if len(y) < 512:
        padding = 512 - len(y)
        y = np.pad(y, (0, padding), mode='constant')

    # Tính MFCC với tham số n_fft nhỏ hơn
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=min(512, len(y)), fmax=sr // 2, n_mels=20)  # Giảm n_mels để tránh cảnh báo
    mfcc = torch.tensor(mfcc.T, dtype=torch.float32)

    # Chuyển thông điệp thành chuỗi nhị phân
    binary_message = ''.join(format(ord(c), '08b') for c in message)
    binary_message += '1111111111111110'  # Bit kết thúc (16 bit)

    # Kiểm tra xem file có đủ chỗ để chứa thông điệp không
    if len(binary_message) > mfcc.size(0):
        raise ValueError("Thông điệp quá dài để giấu trong âm thanh này.")

    # Giấu thông điệp vào trong MFCC
    mfcc_int = mfcc.to(torch.int32)
    for i in range(len(binary_message)):
        mfcc_int[i, 0] = (mfcc_int[i, 0] & 0xFFFE) | int(binary_message[i])  # Thay đổi bit thấp nhất

    mfcc = mfcc_int.to(torch.float32)

    # Khởi tạo mô hình Autoencoder
    model.eval()
    with torch.no_grad():
        encoded_audio = model.encoder(mfcc)
        decoded_audio = model.decoder(encoded_audio)

    # Đảm bảo kích thước đầu ra là dạng NumPy và có kích thước chính xác để lưu
    decoded_audio = decoded_audio.numpy()  # Chuyển tensor thành mảng NumPy
    decoded_audio = decoded_audio.T  # Chuyển chiều về đúng dạng

    # Lưu lại âm thanh đã nhúng thông điệp vào file mới
    sf.write(output_file, decoded_audio, sr)
    print(f"Thông điệp đã được nhúng vào {output_file}")

# Hàm giải mã thông điệp từ âm thanh
def extract_message(audio_file, model):
    # Sử dụng librosa để đọc file âm thanh
    y, sr = librosa.load(audio_file, sr=None)  # Đọc file âm thanh

    # Bổ sung zero-padding nếu tín hiệu quá ngắn
    if len(y) < 512:
        padding = 512 - len(y)
        y = np.pad(y, (0, padding), mode='constant')

    # Tính MFCC với tham số n_fft nhỏ hơn
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=min(512, len(y)), fmax=sr // 2, n_mels=20)  # Giảm n_mels để tránh cảnh báo
    mfcc = torch.tensor(mfcc.T, dtype=torch.float32)

    # Khởi tạo mô hình Autoencoder
    model.eval()
    with torch.no_grad():
        encoded_audio = model.encoder(mfcc)
        decoded_audio = model.decoder(encoded_audio)

    # Giải mã thông điệp từ MFCC
    binary_message = ''
    for i in range(mfcc.size(0)):
        bit = int(decoded_audio[i, 0].item()) & 1  # Lấy bit thấp nhất
        binary_message += str(bit)

    # Tìm bit kết thúc (1111111111111110) để xác định thông điệp
    end_marker = '1111111111111110'
    end_index = binary_message.find(end_marker)
    if end_index == -1:
        print("Không tìm thấy bit kết thúc trong thông điệp.")
        return ""

    binary_message = binary_message[:end_index]

    # Chuyển chuỗi nhị phân thành thông điệp văn bản
    extracted_message = ''.join(
        chr(int(binary_message[i:i+8], 2)) for i in range(0, len(binary_message), 8)
    )

    print(f"Thông điệp đã được giải mã: {extracted_message}")
    return extracted_message

# Khởi tạo mô hình Autoencoder
model = AudioAutoencoder()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Huấn luyện mô hình với dữ liệu âm thanh giả lập
data = torch.randn(32, 13)  # Giả lập dữ liệu MFCC
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data)  # Tính loss
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Ví dụ sử dụng
input_file = "input.wav"  # Đường dẫn đúng đến file âm thanh
output_file = "output_audio_with_message.wav"
message = input("Nhập thông điệp cần giấu vào âm thanh: ")

embed_message(input_file, message, model, output_file)
extracted_message = extract_message(output_file, model)
print(f"Thông điệp giải mã: {extracted_message}")
