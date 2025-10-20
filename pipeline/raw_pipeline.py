import xarray as xr
import os
import sys
import glob
import json
import hashlib

from collections import Counter, defaultdict
import logging
from pyesgf.search import SearchConnection
import requests
from copy import copy
import pytz
import datetime as dt
import time

import dask.array as da

# configurações de modelos e variáveis
from config.variables_config import map_variaveis_meta
from config.models_config import map_modelos_dir
from transformations.temporal import add_time_features_teste_freqs

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
def compute_raw_metrics(list_nc_files, input_model, input_experiment_id, input_variable_id, frequency, output_metrics_path, start_time, end_time):
    """
    calcula métricas da camada raw a partir da lista de arquivos NetCDF baixados

    """
    if not list_nc_files:
        logger_ingestion.warning("nenhum arquivo disponível para cálculo de métricas na camada raw")
        return

    # extraindo os caminhos dos arquivos
    file_paths = [f["path"] for f in list_nc_files if isinstance(f, dict) and "path" in f]

    # abre todos os arquivos em lazy loading
    #ds = xr.open_mfdataset(list_nc_files, engine="h5netcdf", combine="by_coords", parallel=True, chunks={"time": 50})
    ds = xr.open_mfdataset(file_paths, engine="h5netcdf", combine="by_coords", parallel=True, chunks={"time": 50})
    array_dask = ds[input_variable_id]

    # converte para dask dataframe e adiciona features de tempo
    ddf = array_dask.to_dask_dataframe().reset_index()
    ddf = add_time_features_teste_freqs(ddf, frequency=frequency)

    # calcula estatísticas usando Dask
    #vals = array_dask.flatten()
    vals = array_dask.data.ravel()
    vals = vals[~da.isnan(vals)]

    metrics = {
        "time": {
            "start": str(ds["time"].values[0]),
            "end": str(ds["time"].values[-1]),
            "n_steps": len(ds["time"]),
            "frequency": frequency
        },
        "space": {
            "n_lat": ds.sizes.get("lat", None),
            "n_lon": ds.sizes.get("lon", None),
            "n_gridpoints": ds.sizes.get("lat", 0) * ds.sizes.get("lon", 0)
        },
        "stats": {
            "mean": float(da.mean(vals).compute()),
            "std": float(da.std(vals).compute()),
            "min": float(da.min(vals).compute()),
            "max": float(da.max(vals).compute()),
            "percentiles": {
                "p1": float(da.percentile(vals, 1).compute()),
                "p50": float(da.percentile(vals, 50).compute()),
                "p99": float(da.percentile(vals, 99).compute())
            }
        },
        "parquet": {
            "n_rows": int(ddf.shape[0].compute()),
            "n_partitions": ddf.npartitions
        },
        "meta": {
            "model": input_model,
            "experiment": input_experiment_id,
            "variable": input_variable_id
        },
        "execution_time_seconds": end_time - start_time
    }

    # salvar métricas
    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
    with open(output_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger_ingestion.info(f"métricas da camada raw salvas em {output_metrics_path}")
####################
def compute_checksum(file_path, algorithm="sha256", chunk_size=8192):
    """
    calcula o checksum de um arquivo

    """
    
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

####################
def process_variable_raw_teste_freqs(input_model,
                         input_experiment_id,
                         input_variable_id,
                         input_variant_label,
                         output_dir,
                         table_id="Amon",
                         frequency="mon",
                         year_min=None,
                         year_max=None,
                         year_list=None):
    """
    ingestão da variável climática na camada RAW (CMIP6 via ESGF).

    Args:
        input_model (str): Modelo climático (ex.: "EC-Earth3").
        input_experiment_id (str): Experimento (ex.: "historical").
        input_variable_id (str): Variável (ex.: "tas").
        input_variant_label (str): Variante (ex.: "r1i1p1f1").
        output_dir (str): Diretório base da camada RAW.
        table_id (str): Tabela (ex.: "Amon" para mensal, "day" para diário).
        frequency (str): Frequência temporal ("mon", "day").
        year_min, year_max (int): Filtros opcionais por ano inicial/final.
        year_list (list[int]): Lista explícita de anos a baixar.

    Estrutura de saída:
        raw/<model>/<variable>/exp=<experiment>/freq=<frequency>/year=<YYYY>/<file>.nc
        raw/<model>/<variable>/exp=<experiment>/manifest.json
    """

    ####################
    # função auxiliar para normalizar a informação de período de tempo
    def _format_period(year_min=None, year_max=None, year_list=None):
        """
        retorna uma string representando o período temporal usado
        """
        
        if year_list:
            if len(year_list) == 1:
                return f"{year_list[0]}"
            return f"{min(year_list)}-{max(year_list)}"
        elif year_min and year_max:
            return f"{year_min}-{year_max}"
        elif year_min:
            return f"{year_min}-end"
        elif year_max:
            return f"start-{year_max}"
        else:
            return "all"
    ####################

    # início da contagem de tempo
    start_time = time.time()

    logger_read.info(f"inicializando pipeline | raw")
    logger_read.info(f"config: model={input_model}, exp={input_experiment_id}, var={input_variable_id}, table={table_id}, freq={frequency}")

    # conexão com ESGF
    conn = SearchConnection("https://esgf-data.dkrz.de/esg-search", distrib=False)
    try:
        ctx = conn.new_context(
            project="CMIP6",
            source_id=input_model,
            experiment_id=input_experiment_id,
            variable_id=input_variable_id,
            table_id=table_id,
            frequency=frequency,
            variant_label=input_variant_label,
        )

        if ctx.hit_count == 0:
            logger_read.warning("nenhum dataset encontrado no ESGF com esses filtros")
            return

        result = ctx.search()[0]
        files = result.file_context().search()
        logger_read.info(f"dataset encontrado | total de partições = {len(files)}")

    except Exception as e:
        logger_read.error(f"erro de leitura da variável {input_variable_id} | modelo {input_model} | {input_experiment_id}:\n{e}")
        return

    ####################
    # ingestão dos arquivos
    list_nc_files = []
    decadas = defaultdict(int)

    for dataset_file in files:
        file_name = dataset_file.json["title"]
        url = dataset_file.download_url

        # extrair período do arquivo (YYYYMM-YYYYMM ou YYYY-YYYY)
        period = file_name.split("_")[-1].replace(".nc", "")
        period_start, period_end = None, None
        try:
            period_start = period.split("-")[0]
            period_end = period.split("-")[1]
            start_year = int(period_start[:4])
            end_year = int(period_end[:4])
        except Exception:
            logger_read.warning(f"não foi possível extrair anos de {file_name}, pulando")
            continue

        # filtros
        if year_min and end_year < year_min:
            continue
        if year_max and start_year > year_max:
            continue
        if year_list:
            anos_arquivo = set(range(start_year, end_year + 1))
            if not any(ano in anos_arquivo for ano in year_list):
                continue

        decada = (start_year // 10) * 10
        decadas[decada] += 1

        # diretório de saída
        year_dir = os.path.join(
            output_dir,
            #input_model,
            #input_variable_id,
            f"exp={input_experiment_id}",
            f"freq={frequency}",
            f"year={start_year}"
        )
        os.makedirs(year_dir, exist_ok=True)
        file_path = os.path.join(year_dir, file_name)

        logger_read.info(f"baixando arquivo: {file_name}")
        try:
            response = requests.get(url, stream=True)
            with open(file_path, "wb") as f:
                f.write(response.content)

            if os.path.exists(file_path):
                # cálculo do checksum
                checksum = compute_checksum(file_path)

                #list_nc_files.append(file_path)
                list_nc_files.append({"path": file_path, "checksum": checksum})
                
                logger_ingestion.info(f"arquivo salvo em {file_path}")

        except Exception as e:
            logger_ingestion.error(f"erro ao baixar {file_name}: {e}")

    ####################
    # salvar manifesto JSON
    if list_nc_files:
        # tratando o  período de dados para o nome do manifesto
        period_label = _format_period(year_min, year_max, year_list)
        
        # metadados da execução
        manifest = {
            "model": input_model,
            "experiment": input_experiment_id,
            "variable": input_variable_id,
            "variant_label": input_variant_label,
            "table_id": table_id,
            "frequency": frequency,
            "period": period_label,
            #"downloaded_files": [os.path.basename(f) for f in list_nc_files],
            "downloaded_files": [
                                    {
                                        "file_name": os.path.basename(f["path"]),
                                        "checksum_sha256": f["checksum"]
                                    } for f in list_nc_files
                                ],
            "years": sorted(
                                {
                                    int(os.path.basename(f["path"]).split("_")[-1][:4]) for f in list_nc_files
                                }
                            )
        }

        manifest_filename = f"manifest_{input_variable_id}_{input_model.lower()}_{input_experiment_id}_{frequency}_{period_label}.json"

        manifest_path = os.path.join(
            output_dir,
            #input_model,
            #input_variable_id,
            f"exp={input_experiment_id}",
            f"freq={frequency}",
            #f"manifest_{input_variable_id}_{input_model.lower()}_{input_experiment_id}_{frequency}.json"
            manifest_filename
        )
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger_ingestion.info(f"manifesto salvo em {manifest_path}")

    end_time = time.time()

    # métricas
    metrics_filename = f"metrics_raw_{input_variable_id}_{input_model}_{input_experiment_id}_{frequency}_{period_label}.json"

    metrics_path = os.path.join(output_dir,
                                f"exp={input_experiment_id}",
                                f"freq={frequency}",
                                metrics_filename
                            )
    
    compute_raw_metrics(list_nc_files, input_model, input_experiment_id, input_variable_id, frequency, metrics_path, start_time, end_time)
    ####################
    logger_read.info(f"pipeline concluído | raw | arquivos salvos={len(list_nc_files)}")

    #return {"manifest_path": manifest_path, "metrics_path": metrics_path, "n_files": len(list_nc_files)}


'''
def process_variable_raw_old(input_model, input_experiment_id, input_variable_id, input_variant_label, output_dir):
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
            qntd = 10
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
'''

'''
def process_variable_raw_old_teste(input_model, input_experiment_id, input_variable_id, input_variant_label, output_dir):
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
            qntd = 0
            if qntd:
                logger_read.info(f"quantidade de partições que serão processadas: {qntd}")
                files_url_tas_shrunken = files_url_tas[:qntd]
            else:
                logger_read.info(f"todas as partições poderão ser lidas")
                files_url_tas_shrunken = files_url_tas

        else:
            logger_read.info(f"dataset não encontrado")
    except Exception as e:
        logger_read.error(f"erro de leitura da variável {input_variable_id} | modelo {input_model} | {input_experiment_id}:\n{e}")
    ####################
    # ingestão dos arquivos
    list_nc_files = []

    year_min = 1900  # parâmetro configurável

    decadas = defaultdict(int)

    for dataset_tas in files_url_tas_shrunken:

        file_name = dataset_tas.json["title"]
        url = dataset_tas.download_url

        period = file_name.split("_")[-1].replace(".nc", "")
        start_year = int(period.split("-")[0][:4])
        end_year = int(period.split("-")[1][:4])

        # registrar década
        decada = (start_year // 10) * 10
        decadas[decada] += 1

        # aplicar filtro por ano mínimo
        if end_year < year_min:
            logger_read.info(f"pulando {file_name} (termina em {end_year}, antes de {year_min})")
            continue


        logger_read.info(f"arquivo atual = {file_name}")
        
        try:
            response = requests.get(url)

            # extraindo o ano do nome do arquivo
            year_months = file_name.split("_")[-1].replace(".nc", "")
            #year = year_months[:4]
            year = str(start_year)

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

    #logger_read.info(f"\n-----\nquantidade de partições salvas: {len(list_nc_files)}")
    logger_read.info(f"décadas disponíveis no dataset: {sorted(decadas.keys())}")
    ####################
    logger_read.info(f"pipeline concluído | raw")
    ####################
'''

'''
def process_variable_raw(input_model, input_experiment_id, input_variable_id, input_variant_label, output_dir, year_min=None, year_max=None, year_list=None):
    """
    Faz ingestão da variável climática na camada raw com filtros de tempo.

    :param year_min: ano mínimo (ex.: 1900). Se None, ignora.
    :param year_max: ano máximo (ex.: 1920). Se None, ignora.
    :param year_list: lista explícita de anos a baixar (ex.: [1910, 1920, 1930]). Se None, ignora.
    """

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
            qntd = 0
            if qntd:
                logger_read.info(f"quantidade de partições que serão processadas: {qntd}")
                files_url_tas_shrunken = files_url_tas[:qntd]
            else:
                logger_read.info(f"todas as partições poderão ser lidas")
                files_url_tas_shrunken = files_url_tas

        else:
            logger_read.info(f"dataset não encontrado")
    except Exception as e:
        logger_read.error(f"erro de leitura da variável {input_variable_id} | modelo {input_model} | {input_experiment_id}:\n{e}")
    
    ####################
    # ingestão dos arquivos
    list_nc_files = []
    decadas = defaultdict(int)

    for dataset_tas in files_url_tas_shrunken:

        file_name = dataset_tas.json["title"]
        url = dataset_tas.download_url

        # extraindo período do arquivo
        period = file_name.split("_")[-1].replace(".nc", "")
        try:
            start_year = int(period.split("-")[0][:4])
            end_year = int(period.split("-")[1][:4])
        except Exception:
            logger_read.warning(f"não foi possível extrair anos de {file_name}, pulando")
            continue

        # aplicar filtros de ano
        if year_min is not None and end_year < year_min:
            logger_read.info(f"pulando {file_name} (termina em {end_year}, antes de {year_min})")
            continue

        if year_max is not None and start_year > year_max:
            logger_read.info(f"pulando {file_name} (começa em {start_year}, depois de {year_max})")
            continue

        if year_list is not None:
            anos_arquivo = set(range(start_year, end_year + 1))
            if not any(ano in anos_arquivo for ano in year_list):
                logger_read.info(f"pulando {file_name} (fora da lista {year_list})")
                continue

        # contagem de décadas
        decada = (start_year // 10) * 10
        decadas[decada] += 1

        logger_read.info(f"baixando arquivo: {file_name}")

        try:
            response = requests.get(url)

            # criando diretório por ano inicial
            year_dir = os.path.join(output_dir, str(start_year))
            os.makedirs(year_dir, exist_ok=True)

            file_path = os.path.join(year_dir, file_name)

            with open(file_path, "wb") as f:
                f.write(response.content)

            if os.path.exists(file_path):
                list_nc_files.append(file_path)
                logger_read.info(f"arquivo salvo em {file_path}")

        except Exception as e:
            logger_read.error(f"erro ao baixar {file_name}: {e}")
    
    metrics_path = os.path.join(output_dir, "metrics_raw.json")
    
    logger_read.info(f"\n-----\nquantidade final de arquivos salvos: {len(list_nc_files)}")
    logger_read.info(f"décadas disponíveis no dataset: {sorted(decadas.keys())}")
    logger_read.info(f"pipeline concluído | raw")
    ####################
####################
'''