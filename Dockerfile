# Python-ы албан ёсны, хөнгөн хувилбарыг суурь болгоно
FROM python:3.10-slim

# Контейнер дотор ажиллах хавтас үүсгэнэ
WORKDIR /app

# Шаардлагатай сангуудыг суулгах файлыг хуулна
COPY requirements.txt .

# pip ашиглан сангуудыг суулгана
RUN pip install --no-cache-dir -r requirements.txt

# Програмын кодыг контейнер лүү хуулна
COPY ocr_app.py .

# Streamlit-ийн ажиллах портыг нээнэ
EXPOSE 8501

# Контейнерыг ажиллуулахад гүйцэтгэх комманд
# 0.0.0.0 хаяг нь контейнерын гаднаас хандах боломжийг олгоно
CMD ["streamlit", "run", "ocr_app.py", "--server.port=8501", "--server.address=0.0.0.0"]