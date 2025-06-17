# This script is for running evaluations on a base model, before fine-tuning.
#

from utils import yaml_to_munch, create_new_folder, munch_to_yaml
from munch import Munch
from os import path
import argparse
import os
from main import KEYS
import subprocess


def run_base_model_evals(args):
    """
    Runs evaluations on a specified base model using dataset configurations
    from a fine-tuning experiment directory.
    """
    if args.templates is None:
        args.templates = os.listdir(args.template_dir)
        # remove the .yaml extension
        args.templates = [
            t.split(".")[0] for t in args.templates if t.endswith(".yaml")
        ]

    for exp_dir in args.exp_dir:
        for template_name in args.templates:
            assert path.exists(exp_dir), (
                f"Experiment directory {exp_dir} does not exist"
            )

            # Load the experiment's finetune config to get dataset parameters
            finetune_config_path = path.join(exp_dir, "finetune.yaml")
            assert path.exists(finetune_config_path), (
                f"finetune.yaml not found in {exp_dir}"
            )
            finetune_config = yaml_to_munch(finetune_config_path)

            # Load the evaluation template
            template_path = path.join(args.template_dir, f"{template_name}.yaml")
            assert path.exists(template_path), f"Template {template_path} not found"
            template = yaml_to_munch(template_path)

            if "eval" not in template:
                template.eval = Munch()

            # --- Configure the evaluation ---
            # These checks ensure the template is clean before we populate it.
            if "seed" in template.dataset:
                raise ValueError("seed must not be specified in the template")
            if "dataset" in template.eval and template.eval.dataset:
                raise ValueError(
                    "eval.dataset path must not be specified in the template"
                )
            if "model" in template.eval:
                raise ValueError("eval.model must not be specified in the template")

            # 1. Inject the model and seed
            template.eval.model = args.model
            template.dataset.seed = args.seed
            template.eval.dataset = ""  # Ensure we generate the dataset, not load it

            # 2. Copy dataset configuration from the fine-tuning experiment
            if "var_dict" in finetune_config.dataset:
                template.dataset.var_dict = finetune_config.dataset.var_dict
            if "train_functions" in finetune_config.dataset:
                template.dataset.train_functions = (
                    finetune_config.dataset.train_functions
                )
            if "test_functions" in finetune_config.dataset:
                template.dataset.test_functions = finetune_config.dataset.test_functions
            if "system_prompt" in finetune_config.dataset:
                template.dataset.system_prompt = finetune_config.dataset.system_prompt
            if "hide_imports" in finetune_config.dataset:
                template.dataset.hide_imports = finetune_config.dataset.hide_imports

            template.eval.verbose = args.verbose

            # 3. Create a unique directory for this evaluation
            eval_dir_name = f"{template_name}_{args.model}"
            eval_dir = path.join(exp_dir, eval_dir_name)

            try:
                create_new_folder(eval_dir)
            except FileExistsError:
                print(f"Evaluation directory {eval_dir} already exists. Skipping.")
                continue

            # 4. Save the final eval config
            munch_to_yaml(template, path.join(eval_dir, "eval.yaml"))

            # 5. Run the evaluation
            print(f"Running evaluation for {template_name} on model {args.model}")
            print(f"Results will be saved in: {eval_dir}")

            out = subprocess.run(
                [
                    "python",
                    "main.py",
                    "--secrets",
                    args.secrets,
                    "--config",
                    "eval.yaml",
                    "--exp_path",
                    eval_dir,
                    "--task",
                    "eval",
                ]
            )

            assert out.returncode == 0, f"Error running evaluation for {eval_dir}"
            print(f"Finished evaluation for {template_name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluations on a base model before fine-tuning."
    )
    parser.add_argument(
        "--template_dir",
        type=str,
        required=True,
        help="Directory containing evaluation templates.",
    )
    parser.add_argument(
        "--templates",
        nargs="+",
        default=None,
        help="Specific templates to use. If not provided, all templates in the directory will be used.",
    )
    parser.add_argument(
        "--exp_dir",
        nargs="+",
        required=True,
        help="Directory of the fine-tuning experiment, containing finetune.yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the base model to evaluate (e.g., gpt-3.5-turbo-0613).",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed for dataset generation."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during evaluation.",
    )
    parser.add_argument(
        "--secrets", type=str, default=KEYS, help="Path to the file with API keys."
    )
    args = parser.parse_args()

    run_base_model_evals(args)
