import os
import errno
import pytz
import stat
from pipeline.logger import configurar_logger

# configurações de timezone
selected_tz = pytz.timezone("America/Sao_Paulo")

# configura log
logger_dir = configurar_logger("construcao_io", selected_tz)

def ensure_dir(path: str, mode: int = 0o755):
    """
    Garante que o diretório existe e possui permissões adequadas.
    Cria recursivamente se não existir.
    Se existir, ajusta permissões se necessário.
    """
    try:
        os.makedirs(path, mode=mode, exist_ok=True)
        # Garante que o diretório é gravável pelo usuário atual
        current_mode = stat.S_IMODE(os.stat(path).st_mode)
        if current_mode != mode:
            logger_dir.warning(f"ajustando permissões em {path}: {oct(current_mode)} → {oct(mode)}")
            os.chmod(path, mode)
        logger_dir.info(f"diretório garantido: {path}")
    except PermissionError as e:
        logger_dir.error(f"sem permissão para criar/acessar {path}: {e}")
        raise
    except OSError as e:
        if e.errno != errno.EEXIST:
            logger_dir.error(f"erro ao criar diretório {path}: {e}")
            raise