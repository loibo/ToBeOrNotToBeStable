import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--model_name',
                    choices=["nn", "renn", "stnn", "strenn"],
                    help="Select the model you want to test.",
                    type=str,
                    required=True)
parser.add_argument('-ni', '--noise_inj',
                    help="The amount of noise injection. Given as the variance of the Gaussian. Default: 0.",
                    type=float,
                    default=0,
                    required=False)
parser.add_argument('--config',
                    help="The path for the .yml containing the configuration for the model.",
                    type=str,
                    required=False,
                    default=None)
parser.add_argument('--verbose',
                    help="Unable/Disable verbose for the code.",
                    type=str,
                    required=False,
                    default="1",
                    choices=["0", "1"])
args = parser.parse_args()

if args.config is None:
    suffix = str(args.noise_inj).split('.')[-1]
    args.config = f"./config/GoPro_{suffix}.yml"

with open(args.config, 'r') as file:
    setup = yaml.safe_load(file)