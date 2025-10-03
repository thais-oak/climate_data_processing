'''# pipelines
from pipeline.raw_pipeline import process_variable_raw_teste_freqs
from pipeline.trusted_pipeline import process_variable_trusted_dask_only
from pipeline.delivery_pipeline import process_variable_delivery_teste

# metadados variáveis e modelos
from config.variables_config import map_variaveis_meta
from config.models_config import map_modelos_dir

# transformações
from transformations.geo import fix_longitude, select_latam
from transformations.quality import kelvin_to_celsius, remove_outliers

# utils
import os


if __name__ == "__main__":

    dir_raiz = r"/home/thais/climate-ingestion/climate_data_processing/datasets"

    # diretório para CI
    dir_outputs = r"/home/thais/climate-ingestion/climate_data_processing/outputs"

    dir_raw = os.path.join(dir_raiz, "raw")
    #os.makedirs(dir_raw, exist_ok=True)

    dir_trusted = os.path.join(dir_raiz, "trusted")
    #os.makedirs(dir_trusted, exist_ok=True)

    #dir_delivery = os.path.join(dir_raiz, "delivery")
    # teste para CI
    dir_delivery = os.path.join(dir_outputs, "delivery")

    # selecionando a variável
    variavel_escolhida = map_variaveis_meta["tas"]

    # selecionando o modelo
    modelo_escolhido = map_modelos_dir["EC-Earth3"]

    # diretório para os datasets nas camadas
    dir_modelo = modelo_escolhido["nome_dir"]

    # diretório para os datasets | raw
    dir_modelo_raw = os.path.join(dir_raw, dir_modelo, variavel_escolhida["variable_id"])
    os.makedirs(dir_modelo_raw, exist_ok=True)

    # diretório para os datasets | trusted
    dir_modelo_trusted = os.path.join(dir_trusted, dir_modelo, variavel_escolhida["variable_id"])
    os.makedirs(dir_modelo_trusted, exist_ok=True)

    # diretório para os datasets | delivery
    dir_modelo_delivery = os.path.join(dir_delivery, dir_modelo, variavel_escolhida["variable_id"])
    os.makedirs(dir_modelo_delivery, exist_ok=True)

    # parâmetros para a execução da pipeline raw
    source_id = modelo_escolhido["nome"]
    experiment_id = variavel_escolhida["experiment_id"]
    variable_id=variavel_escolhida["variable_id"]#"tas",
    input_table_id = "Amon"
    input_frequency = "mon"
    variant_label = variavel_escolhida["variant_label"]#"r110i1p1f1"

    print(dir_modelo_delivery)
    # parâmetros para a execução da pipeline trusted
    # mapeando nomes das funções para objetos Python
    transform_funcs = {
        "fix_longitude": fix_longitude,
        "select_latam": select_latam,
        "kelvin_to_celsius": kelvin_to_celsius,
        "remove_outliers": remove_outliers
    }

    ##########
    
    import json
    from pathlib import Path

    def build_output_paths(model, experiment, variable, frequency):
        base_dir = Path("outputs") / "delivery" / model / experiment / variable / frequency
        parquet_dir = base_dir / "parquet"
        metrics_dir = base_dir / "metrics"
        
        parquet_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            "parquet": parquet_dir,
            "metrics": metrics_dir
        }
    
    paths = build_output_paths(dir_modelo, experiment_id, variable_id, input_frequency)
    
    ##########

    # execução pipeline raw
    process_variable_raw_teste_freqs(source_id, experiment_id, variable_id, variant_label, dir_modelo_raw, input_table_id, input_frequency, year_min=None, year_max=None, year_list=[1910, 1911])

    # execução pipeline trusted
    #process_variable_trusted_dask_only(source_id, experiment_id, variable_id, variant_label, dir_modelo_raw, dir_modelo_trusted, transform_funcs, input_table_id)

    # variable_id, input_model, input_experiment_id, input_dir, output_dir, grid_step=2.0, apply_pca_flag=False, apply_lasso_flag=False, lasso_target_strategy="mean", n_pca_components=3, lasso_regularization=0.1
    # execução pipeline delivery
    #process_variable_delivery_teste(variable_id, source_id, experiment_id, dir_modelo_trusted, dir_modelo_delivery, grid_step=2.0, apply_lasso_flag=True, apply_pca_flag=True, lasso_target_strategy="mean")
'''
##########

# main_pipeline.py

import os
import argparse
from functools import partial

# pipelines
from pipeline.raw_pipeline import process_variable_raw_teste_freqs
from pipeline.trusted_pipeline import process_variable_trusted_dask_only
from pipeline.delivery_pipeline import process_variable_delivery_teste

# metadados
from config.variables_config import map_variaveis_meta
from config.models_config import map_modelos_dir

# transformações
from transformations.geo import fix_longitude, select_latam
from transformations.quality import kelvin_to_celsius, remove_outliers


if __name__ == "__main__":

    # parser de argumentos (permite escolher a camada)
    parser = argparse.ArgumentParser(description="Climate Data Processing - RAW, TRUSTED, DELIVERY")
    parser.add_argument("--stage", choices=["raw", "trusted", "delivery", "all"], default="all",
                        help="Qual camada executar")
    args = parser.parse_args()

    # raiz dos dados
    dir_raiz = r"/home/thais/climate-ingestion/climate_data_processing/datasets"
    dir_outputs = r"/home/thais/climate-ingestion/climate_data_processing/outputs"

    # diretórios das camadas
    dir_raw = os.path.join(dir_raiz, "raw")
    dir_trusted = os.path.join(dir_raiz, "trusted")
    dir_delivery = os.path.join(dir_outputs, "delivery")  # CI: joga saída em outputs/delivery

    # variável e modelo escolhidos
    variavel_escolhida = map_variaveis_meta["tas"]
    modelo_escolhido = map_modelos_dir["EC-Earth3"]

    dir_modelo = modelo_escolhido["nome_dir"]
    dir_modelo_raw = os.path.join(dir_raw, dir_modelo, variavel_escolhida["variable_id"])
    dir_modelo_trusted = os.path.join(dir_trusted, dir_modelo, variavel_escolhida["variable_id"])
    dir_modelo_delivery = os.path.join(dir_delivery, dir_modelo, variavel_escolhida["variable_id"])

    os.makedirs(dir_modelo_raw, exist_ok=True)
    os.makedirs(dir_modelo_trusted, exist_ok=True)
    os.makedirs(dir_modelo_delivery, exist_ok=True)

    # mapeando transformações
    transform_funcs = {
        "fix_longitude": fix_longitude,
        "select_latam": select_latam,
        "kelvin_to_celsius": kelvin_to_celsius,
        "remove_outliers": remove_outliers
    }

    # dicionário de stages
    stages = {
        "raw": partial(
            process_variable_raw_teste_freqs,
            modelo_escolhido["nome"],                     # source_id
            variavel_escolhida["experiment_id"],          # experiment
            variavel_escolhida["variable_id"],            # variable
            variavel_escolhida["variant_label"],          # variant_label
            dir_modelo_raw,                               # output_dir
            "Amon",                                       # table_id
            "mon",                                        # frequency
            year_list=[1910, 1911]                        # restrição CI/testes
        ),
        "trusted": partial(
            process_variable_trusted_dask_only,
            input_model=dir_modelo,
            input_experiment_id=variavel_escolhida["experiment_id"],
            input_variable_id=variavel_escolhida["variable_id"],
            input_variant_label=variavel_escolhida["variant_label"],
            input_dir=dir_modelo_raw,
            output_dir=dir_modelo_trusted,
            map_transform_funcs=transform_funcs,
            table_id="Amon"
        ),
        "delivery": partial(
            process_variable_delivery_teste,
            dir_modelo_trusted,
            dir_modelo_delivery,
            variavel_escolhida["variable_id"]
        )
    }

    # execução
    if args.stage == "all":
        for stage in ["raw", "trusted", "delivery"]:
            print(f"\n>>> Executando estágio: {stage.upper()} <<<")
            stages[stage]()
    else:
        print(f"\n>>> Executando estágio: {args.stage.upper()} <<<")
        stages[args.stage]()