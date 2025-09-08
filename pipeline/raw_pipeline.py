import xarray as xr
import os
import sys
import glob

from collections import Counter, defaultdict
import logging
from pyesgf.search import SearchConnection
import requests
from copy import copy
import pytz
import datetime as dt

# configurações de modelos e variáveis
from config.variables_config import map_variaveis_meta
from config.models_config import map_modelos_dir

selected_tz = pytz.timezone("America/Sao_Paulo")
####################
# configurações de log
class TZFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.tz = tz or pytz.UTC

    def formatTime(self, record, datefmt=None):
        date_time = dt.datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            s = date_time.strftime(datefmt)
        else:
            s = date_time.isoformat()
        return s

# função de configuração do log
def configurar_logger(nome_logger):
    logger = logging.getLogger(nome_logger)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # evita envio ao root logger

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        # Timezone Brasil (Horário de Brasília)
        #tz_brasil = pytz.timezone("America/Sao_Paulo")

        formatter = TZFormatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S %z',
            tz=selected_tz
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger_read = configurar_logger("leitura_dados")
logger_ingestion = configurar_logger("ingestao_dados")
####################
def process_variable_raw(input_model, input_experiment_id, input_variable_id, input_variant_label, output_dir):
    ####################
    logger_read.info(f"inicializando pipeline | raw")
    # conexão com ESGF e busca dos datasets
    conn = SearchConnection("https://esgf-data.dkrz.de/esg-search", distrib=False)
    try:
        logger_read.info(f"iniciando leitura da variável {input_variable_id} | modelo {input_model} | {input_experiment_id}")

        ctx = conn.new_context(
            project="CMIP6",
            source_id=input_model,#modelo_escolhido["nome"],#"EC-Earth3",
            experiment_id=input_experiment_id,#variavel_escolhida["experiment_id"],#"historical",
            variable_id=input_variable_id,#variavel_escolhida["variable_id"],#"tas",
            table_id="Amon",
            frequency="mon",
            variant_label=input_variant_label,#variavel_escolhida["variant_label"]#"r110i1p1f1"
        )

        if ctx.hit_count:
            logger_read.info(f"dataset encontrado")

            # listagem das URLs dos arquivos do dataset tas | EC-Earth3 | historical | Amon | mon | r110i1p1f1
            result = ctx.search()[0]
            result.dataset_id
            files_url_tas = []

            files = result.file_context().search()
            for file in files:
                #print(file.opendap_url)
                files_url_tas.append(file)

            logger_read.info(f"quantidade de partições do dataset = {len(files_url_tas)}")
            #print(f"quantidade de partições do dataset = {len(files_url_tas)}")

            # lista reduzida de datasets
            qntd = 5
            logger_read.info(f"quantidade de partições que serão processadas: {qntd}")
            files_url_tas_shrunken = files_url_tas[:qntd]

        else:
            logger_read.info(f"dataset não encontrado")
    except Exception as e:
        logger_read.error(f"erro de leitura da variável {input_variable_id} | modelo {input_model} | {input_experiment_id}:\n{e}")
    ####################
    # ingestão dos arquivos
    list_nc_files = []

    for dataset_tas in files_url_tas_shrunken:

        file_name = dataset_tas.json["title"]
        url = dataset_tas.download_url

        logger_read.info(f"arquivo atual = {file_name}")
        
        try:
            response = requests.get(url)

            # extraindo o ano do nome do arquivo
            #year = file_name.split("_")[1]
            year_months = file_name.split("_")[-1].replace(".nc", "")
            year = year_months[:4]

            # criando diretório para o ano
            year_dir = os.path.join(output_dir, year)
            os.makedirs(year_dir, exist_ok=True)

            # diretório completo do arquivo
            file_path = os.path.join(year_dir, file_name)

            # salvando o arquivo
            with open(file_path, "wb") as f:
                f.write(response.content)

            # verificando se o arquivo foi salvo corretamente
            if os.path.exists(file_path):
                list_nc_files.append(file_path)
                logger_read.info(f"arquivo salvo em {file_path}")

        except Exception as e:
            logger_read.error(f"erro ao baixar {file_name}: {e}")

    #logger_read.info(f"\ntotal arquivos Bronze/Raw salvos: {len(list_nc_files)}")
    logger_read.info(f"\n-----\nquantidade de partições salvas: {len(list_nc_files)}")
    ####################
    logger_read.info(f"pipeline concluído | raw")
    ####################