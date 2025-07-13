# Gunakan image dasar dengan Python dan CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set direktori kerja
WORKDIR /app

# Copy file ke dalam container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file yang dibutuhkan
COPY app.py .
COPY start.sh .

# Beri izin eksekusi ke start.sh
RUN chmod +x start.sh

# Ekspose port yang akan dipakai FastAPI
EXPOSE 8000

# Jalankan aplikasi
CMD ["./start.sh"]
