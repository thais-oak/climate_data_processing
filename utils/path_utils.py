import os

def s3_join(*parts):
    """
    esta função concatena partes em um path S3 válido.
    Exemplo:
        s3_join("meu-bucket", "datasets", "trusted", "EC-Earth3", "tas")
        -> "s3://meu-bucket/datasets/trusted/EC-Earth3/tas"
    """
    base = "/".join(p.strip("/").replace("\\", "/") for p in parts)
    return f"s3://{base}"

def detect_frequency(input_dir):
    """
    Detecta frequência a partir do caminho da pasta ou nome do arquivo.
    """
    if "freq=day" in input_dir:
        return "day"
    elif "freq=mon" in input_dir:
        return "mon"
    else:
        # fallback simples: diário se o arquivo contiver 'day', mensal se 'Amon'
        return "mon"

# tamanho total em MB (somando todos os arquivos parquet)
def get_dir_size_mb(path):
    total_bytes = 0
    for root, _, files in os.walk(path):
        for f in files:
            
            try:
                total_bytes += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
            
    return round(total_bytes / (1024 * 1024), 2)

# tamanho aproximado do dataset
def estimate_dataset_size(ds):
    total_bytes = 0
    for var in ds.data_vars:
        total_bytes += ds[var].nbytes
    return round(total_bytes / (1024 ** 2), 2)  # em MB
