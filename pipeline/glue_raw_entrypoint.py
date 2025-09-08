import argparse
from pipeline.raw_pipeline import process_variable_raw
from utils.path_utils import s3_join

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--base_prefix", default="datasets")
    parser.add_argument("--model", required=True)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--experiment_id", default="historical")
    parser.add_argument("--variant_label", required=True)
    args = parser.parse_args()

    # construindo o path de saída no S3
    dir_raw = s3_join(args.bucket, args.base_prefix, "raw", args.model, args.variable)

    process_variable_raw(
        input_model=args.model,
        input_experiment_id=args.experiment_id,
        input_variable_id=args.variable,
        input_variant_label=args.variant_label,
        output_dir=dir_raw
    )