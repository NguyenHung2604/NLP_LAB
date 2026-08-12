from pathlib import Path

# doi ten nay dung voi ten dataset anh upload len Kaggle
# (xem trong panel "Input" ben phai notebook, duoi /kaggle/input/...)
KAGGLE_DATASET_NAME = "en-vi-transformer"
DATA_DIR = f"/kaggle/input/{KAGGLE_DATASET_NAME}"
WORK_DIR = "/kaggle/working"


def get_config():
    return {
        # === duong dan du lieu: doc tu /kaggle/input (read-only) ===
        "src_lang": "en",
        "tgt_lang": "vi",
        "en_file": f"{DATA_DIR}/tst2012.en",
        "vi_file": f"{DATA_DIR}/tst2012.vi",
        "val_en_file": f"{DATA_DIR}/tst2013.en",
        "val_vi_file": f"{DATA_DIR}/tst2013.vi",

        # === tokenizer: ghi vao /kaggle/working (writable) ===
        "tokenizer_file": WORK_DIR + "/tokenizer_{0}.json",
        "min_frequency": 2,

        # === kien truc model ===
        "seq_len": 160,
        "d_model": 512,
        "N": 6,
        "h": 8,
        "dropout": 0.1,
        "d_ff": 2048,

        # === training - Kaggle co GPU nen tang batch_size len cho nhanh ===
        "batch_size": 32,
        "num_epochs": 20,
        "lr": 1e-4,
        "label_smoothing": 0.1,

        # === checkpoint: ghi vao /kaggle/working (writable) ===
        "model_folder": WORK_DIR + "/weights",
        "model_basename": "tmodel_",
        "preload": "latest",   # de "latest": tu dong load lai checkpoint gan nhat neu co (phong khi session bi ngat)
        "experiment_name": WORK_DIR + "/runs/tmodel",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(model_folder) / model_filename)


def latest_weights_file_path(config):
    model_folder = config["model_folder"]
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
