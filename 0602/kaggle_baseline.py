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
    def __init__(self, essays, scores): 
        self.essays = essays.values 
        self.scores = scores.values  
    def __len__(self): 
        return len(self.essays)
    def __getitem__(self, idx):
        text = str(self.essays[idx]) 
        inputs = tokenizer(text, max_length=cfg.MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt") 
        return {
            "input_ids": inputs["input_ids"].squeeze(), 
            "attention_mask": inputs["attention_mask"].squeeze(), 
            "labels": torch.tensor(self.scores[idx], dtype=torch.float) 
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
data = pd.read_csv("train.csv") 
data["score"] -= 1 

train_texts, val_texts, train_scores, val_scores = train_test_split(data["full_text"], data["score"], test_size=0.1, random_state=42) 
train_dataset = EssayDataset(train_texts, train_scores) 
val_dataset = EssayDataset(val_texts, val_scores) 
train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True) 
val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False) 

# initialize model 
model = DebertaForRegression(cfg.MODEL_NAME).to(cfg.DEVICE)  
optimizer = AdamW(model.parameters(), lr=cfg.LR) 
total_steps = len(train_loader) * cfg.EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps) 

# training function
def train_epoch(model, data_loader, optimizer, device, scheduler): 
    model.train() 
    losses = [] 
    for d in tqdm(data_loader, position=0, leave=True, desc="training"):
        input_ids = d["input_ids"].to(device) 
        attention_mask = d["attention_mask"].to(device) 
        labels = d["labels"].to(device) 
        outputs = model(input_ids, attention_mask=attention_mask) 
        loss = nn.MSELoss()(outputs, labels.unsqueeze(1)) 
        losses.append(loss.item()) 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        scheduler.step() 
    return np.mean(losses) 

# evaluation function focusing on optimizing Cohen's Kappa Score
def eval_model(model, data_loader, device): 
    model.eval() 
    predictions = [] 
    actuals = [] 
    with torch.no_grad(): 
        for d in tqdm(data_loader, position=0, leave=True, desc="validating"):
            input_ids = d["input_ids"].to(device) 
            attention_mask = d["attention_mask"].to(device) 
            labels = d["labels"].to(device) 
            outputs = model(input_ids, attention_mask=attention_mask).squeeze() 
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy()) 
    predictions = np.clip(predictions, 0, 5).round() # assume scores are from 0 to 5 
    kappa_score = cohen_kappa_score(actuals, predictions, weights="quadratic") 
    return kappa_score

# run training and validation
best_kappa = 0
for epoch in range(cfg.EPOCHS): 
    print(f"Epoch {epoch+1}/{cfg.EPOCHS}") 
    train_loss = train_epoch(model, train_loader, optimizer, cfg.DEVICE, scheduler) 
    kappa = eval_model(model, val_loader, cfg.DEVICE) 
    print(f"Train Loss: {train_loss} | Validation Cohen Kappa: {kappa}")
    if kappa > best_kappa: 
        torch.save(model.state_dict(), "best_deberta_240602.bin") 
        best_kappa = kappa 
        print("current best model saved") 

        
print("testing")
print(model.load_state_dict(torch.load("best_deberta_240602.bin")))
print("done!")
