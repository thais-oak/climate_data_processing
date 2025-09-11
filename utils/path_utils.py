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