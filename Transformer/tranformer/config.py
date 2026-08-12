from pathlib import Path

def get_config():
    return {
        # Đường dẫn chạy local
        "src_lang": "en",
        "tgt_lang": "vi",
        "en_file": "tst2012.en",  # Train dataset
        "vi_file": "tst2012.vi",
        "val_en_file": "tst2013.en",
        "val_vi_file": "tst2013.vi",


        # Tokenizer
        "tokenizer_file": "tokenizer_{0}.json", 
        "min_frequency": 2,

        # Transformers
        "seq_len": 160,
        "d_model": 512,
        "N": 6,
        "h": 8,
        "dropout": 0.1,
        "d_ff": 2048,

        # Training
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-4,
        "label_smoothing":0.1,

        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",   # "latest" de load checkpoint gan nhat, hoac None de train tu dau
        "experiment_name": "runs/model", 
    }  

def get_weighs_file_path(config, epoch: str):
    model_folder = config["model_folder"] # Lấy tên thư mục, ví dụ: "weights"
    model_filename = f"{config['model_basename']}{epoch}.pt" # Tạo tên file, ví dụ: "tmodel_05.pt"
    return str(Path(".") / model_folder / model_filename) #./weights/tmodel_05.pt


def latest_weights_file_path(config):
    model_folder = config["model_folder"]
    model_filename = f"{config["model_basename"]}*" # * is wild card
    weight_files = list(Path(model_folder).glob(model_filename))

    if len(weight_files) == 0:
        return None

    weight_files.sort()
    return str(weight_files[-1])
