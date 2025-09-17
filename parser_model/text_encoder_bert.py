from transformers import BertTokenizer, BertModel
import torch

class TextEncoder_BERT:
    def __init__(self):
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.to(self.torch_device)

    def __call__(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        for k in inputs.keys():
            if torch.is_tensor(inputs[k]): inputs[k] = inputs[k].to('cuda')
        with torch.no_grad():
            outputs = self.model(**inputs)
        assert outputs.last_hidden_state.shape[0] == 1
        return outputs.last_hidden_state[0,0,:] # return class token
    
class TextEncoder_TinyBERT:
    def __init__(self):
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', model_max_length=256)
        self.model = BertModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        self.model.to(self.torch_device)

    def __call__(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        for k in inputs.keys():
            if torch.is_tensor(inputs[k]): inputs[k] = inputs[k].to(self.torch_device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        assert outputs.last_hidden_state.shape[0] == 1
        return outputs.last_hidden_state[0,0,:]  # [CLS] token representation

if __name__ == "__main__":
    test = TextEncoder_BERT()
    res = test("test string")
    print(res.shape)