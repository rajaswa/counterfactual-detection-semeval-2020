#IMPORTS
import pandas as pd 
import torch
from torch.utils.data import Dataset

#DATASETS
class dataset_task1(Dataset):
    """Dataset for Counterfactual Detection Binary-Classification
    
    Arguments:
        df {pandas DataFrame} -- pandas dataframe for Counterfactual Detection Binary-Classification
        max_len {int} -- maximum input token sequence length
        tokenizer {transformer tokenizer} -- tokenizer to be used to tokenize input text
    Returns:
        input_tensor {torch.tensor} -- input tokens from input text for transformer model
        attention_mask {torch.tensor} -- attention mask for input tokens for transformer model
        label {torch.tensor} -- 1/0 binary grade
    """    
    def __init__(self, df, max_len, tokenizer):       
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.pad = self.tokenizer.pad_token_id
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        label = torch.tensor(self.df['gold_label'][idx])
        
        text = self.df['sentence'][idx]
        input_ids = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_len)
        input_tensor = torch.tensor(input_ids)
        attention_mask = torch.tensor([1]*len(input_tensor) + [0]*(self.max_len-len(input_tensor)))

        if len(input_tensor)<self.max_len:
            input_tensor = torch.cat((input_tensor, (torch.ones(self.max_len-len(input_tensor))*self.pad).long()))
        elif len(input_tensor)>self.max_len:
            input_tensor = input_tensor[:self.max_len]
        
        return (input_tensor, attention_mask, label)

class dataset_task2(Dataset):
    """Dataset for Antecedent-Consequent Detection Regression
    
    Arguments:
        df {pandas DataFrame} -- pandas dataframe for Counterfactual Detection Binary-Classification
        max_len {int} -- maximum input token sequence length
        tokenizer {transformer tokenizer} -- tokenizer to be used to tokenize input text
    Returns:
        input_tensor {torch.tensor} -- input tokens from input text for transformer model
        attention_mask {torch.tensor} -- attention mask for input tokens for transformer model
        label {torch.tensor} -- the start and end locations of antecedent & consequent spans
        length {torch.tensor} -- character length of input text (to be used to scale up/down the span locations)
    """    
    def __init__(self, df, max_len, tokenizer):
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.pad = self.tokenizer.pad_token_id
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):     
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = self.df['sentence'][idx]
        length = len(text)
        #Ant_start, Ant_end, Cons_start, Cons_end
        label = torch.tensor([self.df['antecedent_startid'][idx]/l, self.df['antecedent_endid'][idx]/l, self.df['consequent_startid'][idx]/l, self.df['consequent_endid'][idx]/l])
        
        input_ids = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_len)
        input_tensor = torch.tensor(input_ids)
        attention_mask = torch.tensor([1]*len(input_tensor) + [0]*(self.max_len-len(input_tensor)))

        if len(input_tensor)<self.max_len:
            input_tensor = torch.cat((input_tensor, (torch.ones(self.max_len-len(input_tensor))*self.pad).long()))
        elif len(input_tensor)>self.max_len:
            input_tensor = input_tensor[:self.max_len]
        
        return (input_tensor, attention_mask, label, torch.tensor(length))