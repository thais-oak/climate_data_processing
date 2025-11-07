# mapeamento de variáveis para testes unitários
# cada variável tem: base esperada, modelo, experimento
TEST_VARIABLES = [
    {"model": "EC-Earth3", "variable": "tas", "experiment": "historical", "base": 280.0},
    {"model": "MPI-ESM1-2-LR", "variable": "tasmin", "experiment": "ssp245", "base": 270.0},
    {"model": "CanESM5", "variable": "pr", "experiment": "ssp585", "base": 10.0},
]