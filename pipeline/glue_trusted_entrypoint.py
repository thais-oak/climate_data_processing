import argparse
from pipeline.trusted_pipeline import process_variable_trusted
from transformations.geo import fix_longitude, select_latam
from transformations.quality import kelvin_to_celsius, remove_outliers
from utils.path_utils import s3_join

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--base_prefix", default="datasets")
    parser.add_argument("--model", required=True)
    parser.add_argument("--variable", required=True)
    args = parser.parse_args()

    # construindo diretórios S3
    dir_raw = s3_join(args.bucket, args.base_prefix, "raw", args.model, args.variable)
    dir_trusted = s3_join(args.bucket, args.base_prefix, "trusted", args.model, args.variable)

    # mapeando nomes das funções para objetos Python
    transform_funcs = {
        "fix_longitude": fix_longitude,
        "select_latam": select_latam,
        "kelvin_to_celsius": kelvin_to_celsius,
        "remove_outliers": remove_outliers,
    }

    process_variable_trusted(
        variable_id=args.variable,
        input_dir=dir_raw,
        output_dir=dir_trusted,
        map_transform_funcs=transform_funcs
    )