# pipelines
from pipeline.raw_pipeline import process_variable_raw, process_variable_raw_teste_freqs
from pipeline.trusted_pipeline import process_variable_trusted, process_variable_trusted_dask_teste, process_variable_trusted_dask_only
from pipeline.delivery_pipeline import process_variable_delivery, process_variable_delivery_teste

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

    dir_raw = os.path.join(dir_raiz, "raw")
    #os.makedirs(dir_raw, exist_ok=True)

    dir_trusted = os.path.join(dir_raiz, "trusted")
    #os.makedirs(dir_trusted, exist_ok=True)

    dir_delivery = os.path.join(dir_raiz, "delivery")

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


    # parâmetros para a execução da pipeline trusted
    # mapeando nomes das funções para objetos Python
    transform_funcs = {
        "fix_longitude": fix_longitude,
        "select_latam": select_latam,
        "kelvin_to_celsius": kelvin_to_celsius,
        "remove_outliers": remove_outliers
    }

    # execução pipeline raw
    #process_variable_raw(source_id, experiment_id, variable_id, variant_label, dir_modelo_raw, year_min=None, year_max=None, year_list=[1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919])
    #process_variable_raw_teste_freqs(source_id, experiment_id, variable_id, variant_label, dir_modelo_raw, input_table_id, input_frequency, year_min=None, year_max=None, year_list=[1910, 1911])

    # execução pipeline trusted
    #process_variable_trusted(variable_id, dir_modelo_raw, dir_modelo_trusted, transform_funcs)
    #process_variable_trusted_dask(variable_id, dir_modelo_raw, dir_modelo_trusted, transform_funcs)
    #process_variable_trusted_dask_teste(source_id, experiment_id, variable_id, variant_label, dir_modelo_raw, dir_modelo_trusted, transform_funcs, input_table_id)
    
    #process_variable_trusted_dask_only(source_id, experiment_id, variable_id, variant_label, dir_modelo_raw, dir_modelo_trusted, transform_funcs, input_table_id)

    # variable_id, input_model, input_experiment_id, input_dir, output_dir, grid_step=2.0, apply_pca_flag=False, apply_lasso_flag=False, lasso_target_strategy="mean", n_pca_components=3, lasso_regularization=0.1
    # execução pipeline delivery
    #process_variable_delivery(variable_id, source_id, experiment_id, dir_modelo_trusted, dir_modelo_delivery, grid_step=2.0, apply_lasso_flag=True, apply_pca_flag=True, lasso_target_strategy="mean")
    process_variable_delivery_teste(variable_id, source_id, experiment_id, dir_modelo_trusted, dir_modelo_delivery, grid_step=2.0, apply_lasso_flag=True, apply_pca_flag=True, lasso_target_strategy="mean")