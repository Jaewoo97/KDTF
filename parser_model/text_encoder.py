import torch
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel


class TextEncoder:
    def __init__(self, model='openai/clip-vit-base-patch32'): # possible model: 'openai/clip-vit-base-patch16', 'openai/clip-vit-large-patch14'
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    
        self.tokenizer = CLIPTokenizer.from_pretrained(model)
        self.text_encoder = CLIPTextModel.from_pretrained(model).to(self.torch_device)
        self.model = CLIPModel.from_pretrained(model).to(self.torch_device)

    def __call__(self, text: str) -> torch.Tensor:
        text_inputs = self.tokenizer(
            text, 
            padding="max_length", 
            return_tensors="pt",
            ).to(self.torch_device)
        # text_embeddings = torch.flatten(self.text_encoder(text_inputs.input_ids.to(self.torch_device))['last_hidden_state'],1,-1)
        text_features = self.model.get_text_features(**text_inputs)
        return text_features
    

if __name__ == "__main__":
    test = TextEncoder()
    emb = test("this is a test sentence")
    print(emb.shape)