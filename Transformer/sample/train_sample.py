import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import BilingualDataset, causal_mask, load_raw_dataset, get_or_build_tokenizer
from model import build_transformer


def get_ds(config):
    # 1) doc du lieu song ngu tu file local (thay the load_dataset("huggingface_id"))
    #    train va validation la 2 file KHAC NHAU (tst2012 vs tst2013) - khong random_split
    #    tu 1 file nua, tranh truong hop model "nhin thay" cau validation luc build tokenizer
    train_ds_raw = load_raw_dataset(config["en_file"], config["vi_file"])
    val_ds_raw = load_raw_dataset(config["val_en_file"], config["val_vi_file"])

    # 2) build (hoac load lai neu da co) tokenizer - chi train tren tap TRAIN,
    #    khong dua du lieu validation vao de tokenizer khong "hoc truoc" tu vung cua tap test
    tokenizer_src = get_or_build_tokenizer(config, train_ds_raw, config["src_lang"])
    tokenizer_tgt = get_or_build_tokenizer(config, train_ds_raw, config["tgt_lang"])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["src_lang"], config["tgt_lang"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["src_lang"], config["tgt_lang"], config["seq_len"])

    # in ra do dai cau lon nhat (ca 2 tap) de kiem tra seq_len co du khong
    max_len_src, max_len_tgt = 0, 0
    for item in train_ds_raw + val_ds_raw:
        max_len_src = max(max_len_src, len(tokenizer_src.encode(item[config["src_lang"]]).ids))
        max_len_tgt = max(max_len_tgt, len(tokenizer_tgt.encode(item[config["tgt_lang"]]).ids))
    print(f"Do dai cau nguon dai nhat: {max_len_src}")
    print(f"Do dai cau dich dai nhat: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)  # batch=1 de de decode/in vi du

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
        N=config["N"],
        h=config["h"],
        dropout=config["dropout"],
        d_ff=config["d_ff"],
    )
    return model


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # chi encode 1 lan, dung lai cho tat ca cac buoc decode
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, num_examples=2):
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in val_dataloader:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            assert encoder_input.size(0) == 1, "Validation dataloader phai co batch_size = 1"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            print_msg("-" * 60)
            print_msg(f"NGUON:   {source_text}")
            print_msg(f"DICH THUC: {target_text}")
            print_msg(f"MODEL DICH: {model_out_text}")

            if count >= num_examples:
                break


def train_model(config):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Dang dung device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    model_filename = (
        latest_weights_file_path(config) if preload == "latest" else (get_weights_file_path(config, preload) if preload else None)
    )
    if model_filename:
        print(f"Dang load lai checkpoint: {model_filename}")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("Khong tim thay checkpoint, train tu dau.")

    # ignore_index = [PAD]: khong tinh loss cho cac vi tri pad
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=config["label_smoothing"]
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache() if device == "cuda" else None
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (B, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (B, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)    # (B, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)    # (B, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask)               # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)                             # (B, seq_len, vocab_size)

            label = batch["label"].to(device)  # (B, seq_len)

            # (B, seq_len, vocab_size) -> (B*seq_len, vocab_size) de dua vao CrossEntropyLoss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # sau moi epoch, chay thu vai vi du de xem model dich nhu the nao
        run_validation(
            model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device,
            lambda msg: batch_iterator.write(msg),
        )

        # luu checkpoint
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )
        print(f"Da luu checkpoint: {model_filename}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
