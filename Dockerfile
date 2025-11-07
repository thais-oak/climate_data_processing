# base image
FROM python:3.12-slim-bookworm

# variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# diretório de trabalho
WORKDIR /app

# atualiza e instala dependências
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    openjdk-17-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

# cria o usuário com mesmo UID/GID do host | usar esta configuração apenas para execução local
RUN useradd -m -u 1000 thais
USER thais

# instalando bibliotecas
RUN pip install --no-cache-dir \
    dask==2025.7.0 \
    esgf-pyclient==0.3.1 \
    h5netcdf==1.6.3 \
    netCDF4==1.7.2 \
    numpy==2.3.1 \
    pandas==2.3.1 \
    pyarrow==21.0.0 \
    pyspark==4.0.0 \
    python-dotenv==1.1.1 \
    pytz==2025.2 \
    PyYAML==6.0.2 \
    requests==2.32.4 \
    scikit-learn==1.7.1 \
    scipy==1.16.1 \
    xarray==2025.7.1

# copia todo o código da raiz do projeto para /app
COPY . /app

# expõe porta para interface do spark
EXPOSE 4040

# comando padrão (overwritten pelo docker-compose)
CMD ["python", "main_pipeline.py"]