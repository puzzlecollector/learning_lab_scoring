import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import numpy as np
from tokenizers import AddedToken
from tqdm.auto import tqdm

class Config:
    MODEL_NAME = "microsoft/deberta-v3-small"
    MAX_LENGTH = 1536
    BATCH_SIZE = 28
    LR = 2e-5
    EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_SPLITS = 5

cfg = Config()
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
custom_tokens = [AddedToken("\n", normalized=False), AddedToken(" "*2, normalized=False)]
tokenizer.add_tokens(custom_tokens)

class EssayDataset(Dataset):
    def __init__(self, essays, scores):
        self.essays = essays.values
        self.scores = scores.values
    def __len__(self):
        return len(self.essays)
    def __getitem__(self, idx):
        text = self.essays[idx]
        inputs = tokenizer(text, max_length=cfg.MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(self.scores[idx], dtype=torch.float)
        }

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
class DebertaForRegression(nn.Module):
    def __init__(self, model_name):
        super(DebertaForRegression, self).__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.deberta.resize_token_embeddings(len(tokenizer))
        self.mean_pooling = MeanPooling() 
        self.regression_head = nn.Linear(self.deberta.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)[0]
        mean_pooled_output = self.mean_pooling(outputs, attention_mask)
        return self.regression_head(mean_pooled_output)
    
def train_epoch(model, data_loader, optimizer, device, scheduler): 
    model.train() 
    losses = [] 
    for idx, d in tqdm(enumerate(data_loader), position=0, leave=True, total=len(data_loader), desc="training"):
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
    predictions = np.clip(predictions, 0, 5).round() 
    kappa_score = cohen_kappa_score(actuals, predictions, weights="quadratic") 
    return kappa_score

predicted_prompts = pd.read_csv("predicted_prompt.csv") 
data = pd.read_csv("train.csv")
data["score"] -= 1
merged_df = pd.merge(predicted_prompts, data, on="essay_id")

groups = merged_df['prompt_name']

group_kfold = GroupKFold(n_splits=cfg.N_SPLITS)

best_models = [] 
best_kappas = [] 
for fold, (train_idx, val_idx) in enumerate(group_kfold.split(merged_df["full_text"], merged_df["score"], groups)):
    print(f"========== validating on fold {fold+1} ===========")    
    #augmented_data = augmented_data[(augmented_data["score"]==5) | (augmented_data["score"]==6)]
    #augmented_full_text = pd.concat([data.iloc[train_idx]["full_text"], augmented_data["full_text"]], ignore_index=True)
    #augmented_scores = pd.concat([data.iloc[train_idx]["score"], augmented_data["score"]], ignore_index=True) 
    #print(augmented_full_text.shape, augmented_scores.shape) 
    train_dataset = EssayDataset(merged_df.iloc[train_idx]["full_text"], merged_df.iloc[train_idx]["score"])
    val_dataset = EssayDataset(merged_df.iloc[val_idx]["full_text"], merged_df.iloc[val_idx]["score"])
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = DebertaForRegression(cfg.MODEL_NAME).to(cfg.DEVICE)
    optimizer = AdamW(model.parameters(), lr=cfg.LR)
    total_steps = len(train_loader) * cfg.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps) 

    best_kappa = 0
    for epoch in range(cfg.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, cfg.DEVICE, scheduler)
        kappa = eval_model(model, val_loader, cfg.DEVICE)
        print(kappa)  
        if fold > 0:
            print("all cv scores:") 
            print(best_kappas) 
        if kappa > best_kappa:
            best_kappa = kappa
            best_model_path = f"0608_{fold}_epoch{epoch}_kappa_{best_kappa}_MeanPooling.bin"
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model for fold {fold} saved with kappa {best_kappa}")

    best_models.append(best_model_path)
    best_kappas.append(best_kappa)

print("Training complete.")
print(best_kappas)
