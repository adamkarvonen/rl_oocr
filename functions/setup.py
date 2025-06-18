from setuptools import setup, find_packages

setup(
    name="inductive-oocr-functions",
    version="0.1",
    py_modules=[
        "utils",
        "main",
        "functions",
        "constants",
        "dataset",
        "function_definitions",
        "input_funcs",
        "logging_setup",
        "tests",
    ],
    packages=find_packages(),
    install_requires=[
        # list your project's dependencies here, e.g.,
        "munch==4.0.0",
        "numpy<2.0",
        "openai==1.87.0",
        "PyYAML==6.0.1",
        "scipy",
        "tiktoken==0.7.0",
        "tqdm",
        "torch==2.7.1",
        "transformers==4.52.4",
        "wandb",
        "datasets",
        "accelerate",
        "trl==0.18.2",
        "ipykernel",
        "peft",
    ],
    # PyPI metadata
)
