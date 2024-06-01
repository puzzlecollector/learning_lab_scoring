import torch 
from torch import nn 
from torch.utils.data import DataLoader, Dataset 
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from tokenizers import AddedToken
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import cohen_kappa_score 
from tqdm.auto import tqdm 

class Config: 
    MODEL_NAME = "microsoft/deberta-v3-small" 
    MAX_LENGTH = 1024
    BATCH_SIZE = 32
    LR = 1e-5 
    EPOCHS = 4 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
cfg = Config() 

# load tokenizer 
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME) 
custom_tokens = [AddedToken("\n", normalized=False), AddedToken(" "*2, normalized=False)]
tokenizer.add_tokens(custom_tokens)
print("successfully added tokens!")

# custom dataset 
class EssayDataset(Dataset): 
    def __init__(self, essays): 
        self.essays = essays.values 
    def __len__(self): 
        return len(self.essays)
    def __getitem__(self, idx):
        text = str(self.essays[idx]) 
        inputs = tokenizer(text, max_length=cfg.MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt") 
        return {
            "input_ids": inputs["input_ids"].squeeze(), 
            "attention_mask": inputs["attention_mask"].squeeze()
        }
    
# custom model with regression head 
class DebertaForRegression(nn.Module):
    def __init__(self, model_name):
        super(DebertaForRegression, self).__init__() 
        self.deberta = AutoModel.from_pretrained(model_name) 
        self.deberta.resize_token_embeddings(len(tokenizer))
        self.regression_head = nn.Linear(self.deberta.config.hidden_size, 1) 
    def forward(self, input_ids, attention_mask): 
        outputs = self.deberta(input_ids, attention_mask=attention_mask) 
        last_hidden_state = outputs.last_hidden_state
        mean_pooled_output = torch.mean(last_hidden_state, dim=1) 
        return self.regression_head(mean_pooled_output) 
    
# load data and prepare datasets 
test_data = pd.read_csv("test.csv") 
test_dataset = EssayDataset(test_data["full_text"])
test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False) 

model = DebertaForRegression(cfg.MODEL_NAME).to(cfg.DEVICE) 
model.load_state_dict(torch.load("best_deberta_240602.bin")) 
model.eval() 

# perform inference
predictions = [] 
with torch.no_grad():
    for d in test_loader:
        input_ids = d["input_ids"].to(cfg.DEVICE) 
        attention_mask = d["attention_mask"].to(cfg.DEVICE) 
        outputs = model(input_ids, attention_mask=attention_mask).squeeze()
        predictions.extend(outputs.cpu().numpy()) 
        
predictions = (np.clip(predictions, 0, 5) + 1).round() 

submission = pd.DataFrame({"essay_id": test_data["essay_id"], "predicted_score": predictions})

submission.to_csv("submission_240602.csv", index=False) 

print(submission) # check submission file 
