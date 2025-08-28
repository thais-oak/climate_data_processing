# pipelines
from pipeline.trusted_pipeline import process_variable_trusted
from pipeline.raw_pipeline import process_variable_raw

# metadados variáveis e modelos
from config.variables_config import map_variaveis_meta
from config.models_config import map_modelos_dir

# transformações
from transformations.geo import fix_longitude, select_latam
from transformations.quality import kelvin_to_celsius, remove_outliers
from transformations.temporal import add_time_features

# utils
import os


if __name__ == "__main__":

    dir_raiz = r"/home/thais/climate-ingestion/climate_data_processing/datasets"

    dir_raw = os.path.join(dir_raiz, "raw")
    #os.makedirs(dir_raw, exist_ok=True)

    dir_trusted = os.path.join(dir_raiz, "trusted")
    #os.makedirs(dir_trusted, exist_ok=True)

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

    # parâmetros para a execução da pipeline raw
    source_id = modelo_escolhido["nome"]
    experiment_id = variavel_escolhida["experiment_id"]
    variable_id=variavel_escolhida["variable_id"]#"tas",
    table_id = "Amon"
    frequency = "mon"
    variant_label = variavel_escolhida["variant_label"]#"r110i1p1f1"


    # parâmetros para a execução da pipeline trusted
    # mapeando nomes das funções para objetos Python
    transform_funcs = {
        "fix_longitude": fix_longitude,
        "select_latam": select_latam,
        "kelvin_to_celsius": kelvin_to_celsius,
        "remove_outliers": remove_outliers
}

    #variable = "tas"
    #model = "EC-Earth3"

    #dir_modelo_raw = f"/home/thais/climate-ingestion/climate_data_processing/datasets/raw/{model}/{variable}"
    #dir_modelo_trusted = f"/home/thais/climate-ingestion/climate_data_processing/datasets/trusted/{model}/{variable}"

    #os.makedirs(dir_modelo_raw, exist_ok=True)
    #os.makedirs(dir_modelo_trusted, exist_ok=True)

    # execução pipeline raw
    #process_variable_raw(source_id, experiment_id, variable_id, variant_label, dir_modelo_raw)
    
    # execução pipeline trusted
    process_variable_trusted(variable_id, dir_modelo_raw, dir_modelo_trusted, transform_funcs)
