# Pipeline de Ingestão, Processamento e Redução de Dados Climáticos (CMIP6)

Este repositório contém um pipeline para ingestão, processamento e redução de dados climáticos provenientes do repositório **ESGF** (Earth System Grid Federation), com foco em variáveis atmosféricas simuladas por modelos do projeto **CMIP6**.  

O fluxo contempla desde a padronização dos dados até a redução utilizando **PCA** (redução de dimensionalidade) e **LASSO** (seleção de atributos espaciais).

---

## Modelos e Variáveis Suportados

### Modelos CMIP6
- **CanESM5**  
- **EC-Earth3**  
- **MPI-ESM1-2-LR**

### Variáveis Climáticas
- `tas` — Temperatura média do ar  
- `tasmax` — Temperatura máxima do ar  
- `tasmin` — Temperatura mínima do ar  

---

## Arquitetura do Projeto

```text
climate_data_processing/
├── __init__.py
├── .gitignore
├── pyproject.toml
├── README.md
├── .env
├── .env.docker
├── requirements_reduced.txt
├── Dockerfile
│
├── config/
│   ├── __init__.py
│   ├── models_config.py
│   ├── variables_config.py
│
├── transformations/
│   ├── __init__.py
│   ├── geo.py
│   ├── temporal.py
│   ├── quality.py
│
├── pipeline/
│   ├── __init__.py
│   ├── delivery_pipeline.py
│   ├── main_pipeline.py
│   ├── raw_pipeline.py
│   ├── trusted_pipeline.py
│
├── datasets/
│   ├── raw/
│   ├── trusted/
│   └── delivery/
│
├── docker/
│   ├── docker-compose.delivery.yml
│   ├── docker-compose.raw.yml
│   └── docker-compose.trusted.yml
│
├── .github/workflows
│   ├── ci_raw_pipeline.yml
│   └── ci_trusted_pipeline.yml

```
---

## Pré-requisitos

- **Docker** ≥ 24.0  
- **Docker Compose** ≥ 2.0  
- **Python** ≥ 3.9 (opcional, se executar scripts diretamente)  

---

## Inicialização do Ambiente

Clone o repositório:

```bash
git clone https://github.com/<SEU_USUARIO>/climate_data_processing.git
cd climate_data_processing
```

## Execução do Pipeline

### Execução local
Execute o pipeline parametrizando modelo, variável e métodos de redução:

```bash
python -m pipeline.main_pipeline \
    --stage raw --model CanESM5 \
    --variable tas \
    --exp historical \
    --freq mon \
    --year-min 1910 \
    --year-max 1912
```
## Parâmetros Disponíveis

| Parâmetro       | Valores                          | Descrição                         |
|-----------------|---------------------------------|----------------------------------|
| `--model`       | can_esm5, ec_earth3, mpi_esm1_2_lr | Modelo climático                 |
| `--variable`    | tas, tasmax, tasmin             | Variável atmosférica             |
| `--apply_pca`   | < opcional >                     | Ativa a redução PCA              |
| `--apply_lasso` | < opcional >                   | Ativa a seleção LASSO            |

---

## Saídas Geradas

| Camada   | Conteúdo             | Formato | Caminho                  |
|----------|----------------------|---------|--------------------------|
| raw      | Dados brutos         | nc      | `datasets/raw/`          |
| trusted  | Dados padronizados   | parquet | `datasets/trusted/`      |
| delivery | Dados reduzidos      | parquet | `datasets/delivery/`     |
| metrics  | Métricas da execução | json    | `metrics/*.json`         |

---

## Observações sobre os arquivos por modelo

EC-Earth3: arquivos separados por ano (necessário concatenar para séries temporais).

CanESM5 e MPI-ESM1-2-LR: todos os anos em um único arquivo por variável.

## Considerações Técnicas

O pipeline foi implementado com as ferramentas PySpark e Dask em contêineres Docker, garantindo execução local eficiente sem sobrecarregar memória RAM e CPU.

A execução é escalável para clusters em nuvem, permitindo futura migração para AWS, GCP ou HPC.

O escopo atual é limitado a:

- 3 variáveis climáticas
- 3 modelos
- frequência mensal
- experimento historical
- período de 1910–1912
- sem otimização de parâmetros do PCA ou LASSO

## Possíveis Extensões Futuras

- Ajuste fino dos hiperparâmetros do PCA e LASSO
- Inclusão de séries diárias ou subdiárias
- Ampliação para mais modelos do CMIP6
- Execução em cluster gerenciado (EMR, Kubernetes)