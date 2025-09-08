def s3_join(*parts):
    """
    esta função concatena partes em um path S3 válido.
    Exemplo:
        s3_join("meu-bucket", "datasets", "trusted", "EC-Earth3", "tas")
        -> "s3://meu-bucket/datasets/trusted/EC-Earth3/tas"
    """
    base = "/".join(p.strip("/").replace("\\", "/") for p in parts)
    return f"s3://{base}"