from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def read_parallel_lines(path):
    """Doc file .en / .vi (moi dong la 1 cau), tra ve list[str]."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() != ""]
    return lines


def load_raw_dataset(en_path, vi_path):
    """
    Doc song song 2 file .en va .vi (da align theo tung dong, vd tst2012.en / tst2012.vi),
    tra ve list[dict] dang [{"en": "...", "vi": "..."}, ...] - giong format
    ma cac video huggingface hay dung (vd ds['translation']['en']).
    """
    en_lines = read_parallel_lines(en_path)
    vi_lines = read_parallel_lines(vi_path)
    assert len(en_lines) == len(vi_lines), (
        f"So dong khong khop: {len(en_lines)} (en) vs {len(vi_lines)} (vi). "
        "Hai file phai duoc align tung cau mot."
    )
    return [{"en": en, "vi": vi} for en, vi in zip(en_lines, vi_lines)]


def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    Tu build tokenizer WordLevel tu chinh du lieu local (khong can huggingface datasets/hub).
    Neu tokenizer da ton tai (file json) thi load lai, khong train lai tu dau.
    """
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=config.get("min_frequency", 2),
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def causal_mask(size):
    """
    Tao mask tam giac duoi (khong cho decoder nhin thay cac token o tuong lai).
    Tra ve tensor bool shape (1, size, size), True = duoc phep nhin thay.
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # tru di 2 cho [SOS] va [EOS] (encoder), tru 1 cho [SOS] (decoder input)
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(
                f"Cau qua dai so voi seq_len={self.seq_len}. "
                f"Hay tang seq_len hoac loc bot cau dai trong data."
            )

        # encoder input: [SOS] + tokens + [EOS] + [PAD]...
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # decoder input: [SOS] + tokens + [PAD]... (khong co [EOS], vi day la input dua vao model)
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # label: tokens + [EOS] + [PAD]... (day la cai model phai du doan ra - lech 1 buoc so voi decoder_input)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # encoder_mask: chi che cac vi tri [PAD], shape (1, 1, seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # decoder_mask: che [PAD] VA che ca tuong lai (causal), shape (1, seq_len, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
