import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Đọc tệp âm thanh gốc và tệp âm thanh đã nhúng thông điệp
audio_data, sr = sf.read('input_audio.wav', dtype='int16')  # Tệp âm thanh gốc
modified_audio, sr = sf.read('output_audio.wav', dtype='int16')  # Tệp âm thanh sau khi nhúng thông điệp

# Chọn một phần nhỏ của tín hiệu âm thanh (để vẽ biểu đồ dễ dàng hơn)
num_samples = 1000  # Số mẫu cần vẽ biểu đồ (bạn có thể điều chỉnh số này)

# Lấy dữ liệu âm thanh gốc và đã thay đổi trong phạm vi số mẫu đã chọn
original_samples = audio_data[:num_samples]
modified_samples = modified_audio[:num_samples]

# Vẽ biểu đồ so sánh sự thay đổi của các mẫu âm thanh
plt.figure(figsize=(10, 6))

# Biểu đồ 1: Mẫu âm thanh gốc
plt.subplot(2, 1, 1)
plt.plot(original_samples, label='Mẫu âm thanh gốc', color='blue')
plt.title("Mẫu âm thanh gốc")
plt.xlabel("Số mẫu")
plt.ylabel("Giá trị mẫu")
plt.grid(True)
plt.legend()

# Biểu đồ 2: Mẫu âm thanh sau khi nhúng thông điệp
plt.subplot(2, 1, 2)
plt.plot(modified_samples, label='Mẫu âm thanh sau khi nhúng', color='red')
plt.title("Mẫu âm thanh sau khi nhúng thông điệp")
plt.xlabel("Số mẫu")
plt.ylabel("Giá trị mẫu")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Vẽ sự khác biệt giữa các mẫu âm thanh
difference = modified_samples - original_samples

# Vẽ biểu đồ sự khác biệt
plt.figure(figsize=(10, 6))
plt.plot(difference, label='Sự khác biệt giữa âm thanh gốc và âm thanh đã nhúng', color='green')
plt.title("Sự thay đổi giữa âm thanh gốc và âm thanh đã nhúng")
plt.xlabel("Số mẫu")
plt.ylabel("Sự khác biệt")
plt.grid(True)
plt.legend()
plt.show()
