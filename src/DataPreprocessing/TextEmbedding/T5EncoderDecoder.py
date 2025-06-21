import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class T5EncoderDecoder:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("Loading T5 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  # tokenizer = AutoTokenizer.from_pretrained("t5-base")
        print("Loading T5 model...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to(device)

        self.embedding_size = self.tokenizer.model_max_length

    def encode(self, input_string: str, move_tensor_to_device: bool = True) -> torch.Tensor:
        text = "paraphrase: " + input_string + " </s>"
        encoding = self.tokenizer.encode_plus(text, padding="max_length", return_tensors="pt")
        input_ids = encoding["input_ids"][0]

        if move_tensor_to_device:
            input_ids = input_ids.to(self.device)

        return input_ids

    def decode(self, input_ids: torch.Tensor) -> str:
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=self.embedding_size,
            do_sample=True,
            top_k=1,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=1
        )

        encoded_output = outputs[0]
        return self.tokenizer.decode(encoded_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
