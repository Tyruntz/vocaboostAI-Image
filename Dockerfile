# Gunakan base image ringan + PyTorch support
FROM python:3.10-slim

# Buat working directory
WORKDIR /app

# Copy semua file
COPY . .

# Install pip + dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Bikin start.sh bisa dijalankan
RUN chmod +x start.sh

# Expose port
EXPOSE 8000

# Jalankan server
CMD ["./start.sh"]
