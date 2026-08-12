from pathlib import Path


def get_config():
    return {
        # === duong dan du lieu local (thay vi ten dataset tren huggingface) ===
        "src_lang": "en",
        "tgt_lang": "vi",
        "en_file": "tst2012.en",       # du lieu de TRAIN
        "vi_file": "tst2012.vi",
        "val_en_file": "tst2013.en",   # du lieu de VALIDATION (file rieng, khong trung voi tap train)
        "val_vi_file": "tst2013.vi",

        # === tokenizer ===
        "tokenizer_file": "tokenizer_{0}.json",  # se tao tokenizer_en.json, tokenizer_vi.json
        "min_frequency": 2,        # tu xuat hien < 2 lan se bi coi la [UNK] (voi data nho, co the giam xuong 1)

        # === kien truc model (khop voi build_transformer trong model.py) ===
        "seq_len": 160,            # do dai toi da 1 cau (tinh theo token) - IWSLT en-vi co cau kha dai
        "d_model": 512,
        "N": 6,
        "h": 8,
        "dropout": 0.1,
        "d_ff": 2048,

        # === training ===
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-4,
        "label_smoothing": 0.1,

        # === luu checkpoint ===
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,           # "latest" de load checkpoint gan nhat, hoac None de train tu dau
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = config["model_folder"]
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
