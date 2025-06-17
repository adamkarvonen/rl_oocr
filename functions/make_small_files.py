import os


def make_small_jsonl(input_path, output_path, n_lines=20):
    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for i, line in enumerate(fin):
            if i >= n_lines:
                break
            fout.write(line)


base_dir = "dev/047_functions/finetune_01"
make_small_jsonl(
    os.path.join(base_dir, "047_func_01_train_oai.jsonl"),
    os.path.join(base_dir, "047_func_01_train_oai_small.jsonl"),
    n_lines=20,
)
make_small_jsonl(
    os.path.join(base_dir, "047_func_01_test_oai.jsonl"),
    os.path.join(base_dir, "047_func_01_test_oai_small.jsonl"),
    n_lines=5,
)
print("Small files created.")
