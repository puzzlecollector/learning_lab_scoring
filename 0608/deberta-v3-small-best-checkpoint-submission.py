import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import numpy as np
from tokenizers import AddedToken
from tqdm.auto import tqdm

class Config:
    MODEL_NAME = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-small"
    MAX_LENGTH = 1536
    BATCH_SIZE = 16
    LR = 2e-5
    EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_SPLITS = 5

cfg = Config()

tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
custom_tokens = [AddedToken("\n", normalized=False), AddedToken(" "*2, normalized=False)]
tokenizer.add_tokens(custom_tokens)

class EssayDataset(Dataset):
    def __init__(self, essays):
        self.essays = essays.values
    def __len__(self):
        return len(self.essays)
    def __getitem__(self, idx):
        text = self.essays[idx]
        inputs = tokenizer(text, max_length=cfg.MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze()
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

test_data = pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv")
test_dataset = EssayDataset(test_data["full_text"])
test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

model_paths = ["/kaggle/input/augmented-folds-deberta-small-0608/Augmented_MSE_DeBERTa-v3-small_0_epoch3_kappa_0.8280886499882285.bin", 
               "/kaggle/input/240604-deberta-small-voting/240604_deberta_small_1.bin", 
               "/kaggle/input/240604-deberta-small-voting/240604_deberta_small_2.bin", 
               "/kaggle/input/augmented-folds-deberta-small-0608/Augmented_MSE_DeBERTa-v3-small_3_epoch2_kappa_0.8261538079143165.bin",
               "/kaggle/input/240604-deberta-small-voting/240604_deberta_small_4.bin"]

model = DebertaForRegression(cfg.MODEL_NAME).to(cfg.DEVICE)

all_predictions = []
with torch.no_grad():
    for model_path in model_paths:
        print(model.load_state_dict(torch.load(model_path)))
        model.eval()
        fold_predictions = []
        for batch in test_loader:
            input_ids = batch['input_ids'].to(cfg.DEVICE)
            attention_mask = batch['attention_mask'].to(cfg.DEVICE)
            outputs = model(input_ids, attention_mask).squeeze()
            fold_predictions.extend(outputs.cpu().numpy())
        all_predictions.append(fold_predictions)
        
all_predictions = np.array(all_predictions)

# print(all_predictions) 

mean_predictions = [] 
for i in range(all_predictions.shape[1]): 
    s = 0 
    for j in range(all_predictions.shape[0]):
        s += all_predictions[j, i] 
    s /= all_predictions.shape[0] 
    mean_predictions.append(s)
        
mean_predictions = (np.clip(mean_predictions, 0, 5) + 1).round()

submission = pd.DataFrame({
    "essay_id": test_data["essay_id"],
    "score": mean_predictions
})

submission.score = submission.score.astype('int32')

submission.to_csv("submission.csv", index=False)

print(submission) 
