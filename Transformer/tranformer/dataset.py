from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# doc mot loat dong mot luc
def read_parallel_lines(path):
    # utf-8 la bang ma dung cho tieng viet
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() != ""]
        # strip() su dung de loai bo \n va \t dau dong va cuoi dong, 
        # voi nhung dong khong phai dong trong (line.strip() != "")
    return lines


def load_raw_dataset(en_path, vn_path):
    en_lines = read_parallel_lines(en_path)
    vn_lines = read_parallel_lines(vn_path)

    assert len(en_lines) == len(vn_lines) , (
        "Số lượng dòng của hai bản text phải khớp với nhau,"
        f"Số lượng dòng không khớp: {len(en_lines)} (en) vs {len(vn_lines)} (vi)"
    )
    return [{"en": en, "vi": vi} for en, vi in zip(en_lines, vn_lines)]

def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang] # moi gia tri se duoc xuat ra moi lan goi ma khong can phai chay het vong lap

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer =  WordLevelTrainer(
            special_tokens= ["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=config.get("min_frequency", 2)
        )   
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True) # dam bao lien ket phai duoc ton tai truoc (lien ket thi co father, ton tai lien ket = oke)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def causal_mask(size):
    """
    Tao mask tam giac duoi (khong cho decoder nhin thay cac token o tuong lai).
    Tra ve tensor bool shape (1, size, size), True = duoc phep nhin thay. (batch size / head dimension, row, col)
    """

    # triu = upper triangle 
    # ma tran toan so 1 ben tren ben phai, diagnonal = 1 == bo luon ca duong cheo tuc duong cheo = 0 het
    mask = torch.triu(torch.ones((1, size, size)), diagonal= 1).type(torch.int)
    return mask == 0 # ci tri token duoc phep nhin thay chuyen thanh true het
class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype= torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype= torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype= torch.int64)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # ds se co dang nhu the nay:
        # {
        #    "en": "Hello, how are you?",       # src_text (câu tiếng Anh)
        #    "vi": "Xin chào, bạn khỏe không?"  # tgt_text (câu tiếng Việt)
        # }
        src_target_pair = self.ds[idx] 
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        # Chuyen hoa thanh tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        # ids: xuat ra  danh sach id cua tokenizer

        # Tính số lượng padding của encoder và decoder
        # encoder input: SOS + tokens cua cau + eos + pad + pad + ...
        # decoder input: SOS + tokens cua cau + pad + pad + pad (khong co eos)

        enc_number_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_number_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_number_padding_tokens < 0 or dec_number_padding_tokens < 0:
            raise ValueError(
                f"Cau qua dai so voi seq_len ={self.seq_len}. "
                f"Hay tang seq_len hoac loc bot cau dai trong data."
            )

        # Encoder input, ghep cac tensor lai voi nhau
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64), # chuyen hoa thanh tensor
                self.eos_token,
                torch.full((enc_number_padding_tokens,), self.pad_token.item(), dtype=torch.int64),
            ],
            dim = 0, # chieu ma se duoc concat voi nhau
        )

        # decoder input: [SOS] + tokens + [PAD]... (khong co [EOS], vi day la input dua vao model)  
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.full((dec_number_padding_tokens,), self.pad_token.item(), dtype=torch.int64),
            ],
            dim = 0,
        )

        # decoder label: tokens + [EOS] + [PAD] + [PAD] + ....
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.full((dec_number_padding_tokens,), self.pad_token.item(), dtype=torch.int64),
                # sua tu tensor thanh full
            ],
            dim = 0,
        )

        # kiem thu noi bo, assertion
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # return ra thanh qua
        return {
            "encoder_input": encoder_input, #(seq_len)
            "decoder_input": decoder_input, #(seq_len)
            # encoder_mask: chi che cac vi tri [PAD], shape (1, 1, seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # decoder_mask: che [PAD] VA che ca tuong lai (causal), shape (1, seq_len, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label, # (seq, len)
            "src_text": src_text,
            "tgt_text": tgt_text,
            
        }