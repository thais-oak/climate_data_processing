# glue_entrypoint.py
import argparse
from pipeline.delivery_pipeline import process_variable_delivery

def str2bool(s):
    return str(s).lower() in {"1","true","t","yes","y"}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--base_prefix", default="datasets")
    p.add_argument("--variable_id", required=True)      # ex.: tas
    p.add_argument("--input_model", required=True)      # ex.: EC-Earth3
    p.add_argument("--experiment_id", required=True)    # ex.: historical
    p.add_argument("--grid_step", type=float, default=2.0)
    p.add_argument("--apply_pca", type=str2bool, default=False)
    p.add_argument("--apply_lasso", type=str2bool, default=False)
    p.add_argument("--lasso_target_strategy", default="mean")
    p.add_argument("--n_pca_components", type=int, default=3)
    p.add_argument("--lasso_regularization", type=float, default=0.1)
    args = p.parse_args()

    base_s3 = f"s3://{args.bucket}/{args.base_prefix}"
    dir_trusted  = f"{base_s3}/trusted/{args.input_model}/{args.variable_id}"
    dir_delivery = f"{base_s3}/delivery"

    process_variable_delivery(
        variable_id=args.variable_id,
        input_model=args.input_model,
        input_experiment_id=args.experiment_id,
        input_dir=dir_trusted,
        output_dir=dir_delivery,
        grid_step=args.grid_step,
        apply_pca_flag=args.apply_pca,
        apply_lasso_flag=args.apply_lasso,
        lasso_target_strategy=args.lasso_target_strategy,
        n_pca_components=args.n_pca_components,
        lasso_regularization=args.lasso_regularization,
    )
