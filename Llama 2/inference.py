from typing import Optional
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from model import ModelArgs, Transformer
import torch  

class LlaMa:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor ,args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    @staticmethod 
    def build(checkpoint_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time() # Danh dau thoi gian xuat phat, sau do tinh thoi gian troi qua bang lay time sau cach tru di moc nay
        if load_model:
            # Chon sorted theo bang chu cai 
            checkpoints = sorted(Path(checkpoint_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoint_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location= "cpu")
            print(f'Loading checkpoint in {-prev_time + time.time():.2f}s')
            prev_time = time.time()

        with open(Path(checkpoint_dir) / "params.json", "r") as f: # cho nay se phai sua lai
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params   
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor) 
            # torch.set_default_tensor_type(torch.float16)
        else: 
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint['rope.freqs'] # ma tran nay khong phai hoc nen khong can cho vao
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loading state dict in {time.time() - prev_time: .2f}s")

        return LlaMa(model,tokenizer,model_args)

    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1 # kich thuoc context window = max so luong gen ra + 1 token moi nhu

        # Convert prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos= False) for prompt in prompts]
        # Make sure that the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"

        max_prompt_len = max(len(prompt) for prompt in prompt_tokens) 
        # Make sure the prompt length is not larger than the maximum sequence length
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        # tao ra cong tac eos_reach voi tung cau
        eos_reached = torch.tensor([False]* batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise (vi pad_id = 0) 
        cur_iterator = tqdm(range(1, total_len), desc= "Generating tokens")


        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos - 1: cur_pos], cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)   
            else: 
                # Greedily select the token with the max probability
                next_token = torch.argmax(logits[:, -1], dim = -1)

            next_token = next_token.reshape(-1) # cán phẳng nó trở thành tensor 1 chiều
            # Only replace token if it is a padding token
            # torch.where(dieu kien, ket qua dung, ket qua sai)
            # kiem tra xem dang la prompt cua nguoi dung hay da het roi
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position
            # toan tu |= la toan tu Or va gan, ~ la toan tu not 
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            # Ngat vong lap for khi toan bo cac cau deu da dat duoc eos
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for _, current_prompt_tokens in enumerate(tokens.tolist()): # prompt index khong can dung
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]


            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
            
        return (out_tokens, out_text)
        
    # def text_completion_speculative(self, draft_model, ):
        
    def _sample_top_p(self, probs, p):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size), cumsum la cumulative sum: ham tich luy
        # vi du: probs_sort = [0.3 0.2 0.1] -> probs_sum = [0.3 0.5 0.6]
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p 
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0  # truyen vao mot mang true/false, sau do thi cai nao true thi cho bang 0
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True)) # div_: chia tai cho cho tong cua probs_sort, -> thanh tong bang 1
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1) # chon ngay nhien mot phan tu torng dong do, xuat ra vi tri
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token) # tim kiem gia tri tai vi tri do
        return next_token
        
        
if __name__ == '__main__':
    torch.manual_seed(0)
    allow_cuda = True # chay tren kaggle 
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    model = LlaMa.build(checkpoint_dir="",
                        tokenizer_path=,
                        load_model=True,
                        max_seq_len=1024,
                        max_batch_size=1,
                        device=)
    

    print("Mo hinh da oke roi dai ca")

    while True:
        user_prompt = input(">> ") # cin
        if user_prompt.strip().lower() == "exit":
            break

        out_tokens, out_texts = model.text_completion([user_prompt], max_gen_len = 64)
        print(out_texts[0]) # trich xuat phan tu dau tien, tu la loi phan tu tu trong hop ra
        print("-" * 50)
    