import argparse
import subprocess
import os
import shutil
import socket

dataset_folder = "datasets"
sample_folder = "samples"
model_folder = "models"
base_model_folder = "YOUR_BASE_MODELS_FOLDER"
test_folder = "tests"

def run(cmd, env=None):
    print("Running:", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, env=env, check=True, preexec_fn=os.setsid)

def main():
    parser = argparse.ArgumentParser(description="Pipeline script with argument parsing.")
    parser.add_argument('--optim', type=str, default="", help='Optimizer string')
    parser.add_argument('--task', type=str, default="", help='Task string')
    parser.add_argument('--gen_temp', type=str, default="1.0", help='Generating temperature')
    parser.add_argument('--train_temp', type=str, default="1.0", help='Training temperature')
    parser.add_argument('--test_temp', type=str, default="0.1", help='Testing temperature')
    parser.add_argument('--model_path', type=str, default="", help='Model path')
    parser.add_argument('--result_prefix', type=str, default="", help='Result prefix')
    parser.add_argument('--all_devices', type=str, default="", help='All CUDA devices')
    parser.add_argument('--n_epoch', type=str, default="0", help='Number of epochs')
    parser.add_argument('--batch_size', type=str, default="32", help='Number of batch size')
    parser.add_argument('--test_data_path', type=str, default="", help='Test data path')
    parser.add_argument('--train_data_path', type=str, default="", help='Raw data path')
    parser.add_argument('--jobs', type=str, default="", help='Names of jobs')
    parser.add_argument('--is_base', type=bool, default=False)
    args = parser.parse_args()

    N_SEQ = "100"
    N_ITER = "1"
    GEN_TEMP = args.gen_temp
    TRAIN_TEMP = args.train_temp
    TEST_TEMP = args.test_temp
    BATCH_SIZE = args.batch_size
    N_CHECK = "1"
    N_EPOCH = args.n_epoch
    EVAL_K = "1,5,10"

    GEN_DATA_PATH = f"{args.train_data_path}/{args.model_path}-{GEN_TEMP}"
    MODEL_SAVE_PATH = f"{model_folder}/{args.train_data_path}/{args.result_prefix}{args.model_path}-{GEN_TEMP}-{TRAIN_TEMP}-{args.optim}-{args.task}-{N_EPOCH}"
    BASE_MODEL_SAVE_PATH = args.model_path
    # BASE_MODEL_SAVE_PATH = f"{base_model_folder}/{args.model_path}"
    TEST_SAVE_PATH = f"{test_folder}/{args.test_data_path}/{args.result_prefix}{args.model_path}-{GEN_TEMP}-{TRAIN_TEMP}-{TEST_TEMP}-{args.optim}-{args.task}-{N_EPOCH}"
    if args.is_base:
        MODEL_SAVE_PATH = f"{base_model_folder}/{args.model_path}"
        TEST_SAVE_PATH = f"{test_folder}/{args.test_data_path}/{args.result_prefix}{args.model_path}-{TEST_TEMP}"
    
    env_all_devices = os.environ.copy()
    env_all_devices["CUDA_VISIBLE_DEVICES"] = args.all_devices

    JOBS = args.jobs.split(',')

    # -----------------------sampling-----------------------
    if 'sampling' in JOBS:
        REGEN = "True"
        run([
            "python", "infer.py",
            "--eval_mode", "False",
            "--regen", REGEN,
            "--data_path", f"{dataset_folder}/{args.train_data_path}",
            "--model_path", BASE_MODEL_SAVE_PATH,
            "--n_seq", N_SEQ,
            "--n_iter", N_ITER,
            "--temp", GEN_TEMP,
            "--save_path", f"{sample_folder}/{GEN_DATA_PATH}"
        ], env=env_all_devices)

    # -----------------------annotating-----------------------
    if 'annotating' in JOBS:
        REGEN = "False"
        run([
            "python", "infer.py",
            "--eval_mode", "False",
            "--regen", REGEN,
            "--data_path", f"{dataset_folder}/{args.train_data_path}",
            "--model_path", BASE_MODEL_SAVE_PATH,
            "--n_seq", N_SEQ,
            "--n_iter", N_ITER,
            "--temp", GEN_TEMP,
            "--save_path", f"{sample_folder}/{GEN_DATA_PATH}",
            "--n_check", N_CHECK
        ])

    # -----------------------merging-----------------------
    if 'merging' in JOBS:
        run([
            "python", "merge.py",
            "--data_path", f"{sample_folder}/{GEN_DATA_PATH}",
            "--save_path", f"{dataset_folder}/{GEN_DATA_PATH}"
        ])

    # -----------------------training-----------------------
    if 'training' in JOBS:
        if os.path.isdir(MODEL_SAVE_PATH):
            shutil.rmtree(MODEL_SAVE_PATH)
            print(f"Deleting folder: {MODEL_SAVE_PATH}")
        else:
            print(f"Will create folder: {MODEL_SAVE_PATH}")

        N_DEVICES = ''.join([c for c in args.all_devices if c.isdigit()])
        DISTRIBUTED_ARGS = [
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=localhost:0",
            "--nnodes=1",
            f"--nproc-per-node={len(N_DEVICES)}"
        ]
        run([
            "torchrun", *DISTRIBUTED_ARGS, "train.py",
            # "python", "train.py",
            "--data_path", f"{dataset_folder}/{GEN_DATA_PATH}",
            "--model_path", BASE_MODEL_SAVE_PATH,
            "--optim", args.optim,
            "--task", args.task,
            "--ds_config", "ds_zero.json",
            "--save_path", MODEL_SAVE_PATH,
            "--batch_size", BATCH_SIZE,
            "--num_epochs", N_EPOCH,
            "--temp", TRAIN_TEMP,
            "--train_data_path", f"{dataset_folder}/{args.train_data_path}"
        ], env=env_all_devices)

    # -----------------------generating-----------------------
    if 'generating' in JOBS:
        REGEN = "True"
        run([
            "python", "infer.py",
            "--eval_mode", "True",
            "--regen", REGEN,
            "--data_path", f"{dataset_folder}/{args.test_data_path}",
            "--model_path", MODEL_SAVE_PATH,
            "--n_seq", N_SEQ,
            "--n_iter", N_ITER,
            "--temp", TEST_TEMP,
            "--save_path", TEST_SAVE_PATH,
            "--eval_k", EVAL_K
        ], env=env_all_devices)

    # -----------------------evaling-----------------------
    if 'evaling' in JOBS:
        REGEN = "False"
        run([
            "python", "infer.py",
            "--eval_mode", "True",
            "--regen", REGEN,
            "--data_path", f"{dataset_folder}/{args.test_data_path}",
            "--n_seq", N_SEQ,
            "--n_iter", N_ITER,
            "--temp", TEST_TEMP,
            "--save_path", TEST_SAVE_PATH,
            "--n_check", N_CHECK,
            "--eval_k", EVAL_K
        ])

if __name__ == "__main__":
    # Print current machine's real IP address
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
        s.connect(('8.8.8.8', 80))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = '127.0.0.1'
    finally:
        s.close()
    print(f"Current machine's IP address: {ip_address}")

    main()