# glue_delivery_entrypoint.py
import argparse
import os
from pipeline.delivery_pipeline import process_variable_delivery

def main():
    parser = argparse.ArgumentParser(description="AWS Glue Delivery Pipeline")

    # parâmetros obrigatórios
    parser.add_argument("--variable_id", type=str, required=True, help="Variável climática (ex: tas)")
    parser.add_argument("--model_id", type=str, required=True, help="Modelo climático (ex: EC-Earth3)")
    parser.add_argument("--experiment_id", type=str, required=True, help="Experimento (ex: historical)")
    parser.add_argument("--dir_trusted", type=str, required=True, help="Diretório da camada trusted (S3)")
    parser.add_argument("--dir_delivery", type=str, required=True, help="Diretório de saída delivery (S3)")

    # parâmetros opcionais
    parser.add_argument("--grid_step", type=float, default=2.0, help="Resolução espacial")
    parser.add_argument("--apply_pca", action="store_true", help="Aplica PCA")
    parser.add_argument("--apply_lasso", action="store_true", help="Aplica LASSO")
    parser.add_argument("--lasso_target_strategy", type=str, default="mean", choices=["mean", "max", "min"],
                        help="Estratégia de agregação para target do LASSO")
    parser.add_argument("--n_pca_components", type=int, default=3, help="Número de componentes principais para PCA")
    parser.add_argument("--lasso_regularization", type=float, default=0.1, help="Parâmetro de regularização LASSO")

    args = parser.parse_args()

    # chama pipeline de delivery
    process_variable_delivery(
        variable_id=args.variable_id,
        input_model=args.model_id,
        input_experiment_id=args.experiment_id,
        input_dir=args.dir_trusted,
        output_dir=args.dir_delivery,
        grid_step=args.grid_step,
        apply_pca_flag=args.apply_pca,
        apply_lasso_flag=args.apply_lasso,
        lasso_target_strategy=args.lasso_target_strategy,
        n_pca_components=args.n_pca_components,
        lasso_regularization=args.lasso_regularization
    )

if __name__ == "__main__":
    main()