from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    
        #init
        def __init__(self, data_path, tokenizer, max_length=512):
            super().__init__()
            self.tokenizer = tokenizer
            self.max_length = max_length #输给GPU的最大长度
            # 使用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
            self.samples = load_dataset("json", data_files=data_path, split="train")

        #__len__
        def __len__(self):
            return len(self.samples)
        #__getitem__我们要拿到的是jsonl里的每一行
        #要输出的是id，attention_mask，labels
        def __getitem__(self,index):
            sample=self.samples[index] #分出样本
            #tokenizer将文本转化为input_id
            tokens = self.tokenizer(
                str(sample["text"])
                add_special_tokens=False
                max_length=self.max_length-2
                truncation=True
            ).input_ids
            #添加上bos eos还有pad填充
            tokens=[self.tokenizer.bos_token_id]+tokens+[self.tokenizer.eos_token_id]
            input_ids=tokens+[self.tokenizer.pad_token_id]*(self.max_length-len(tokens))#填充到固定长度，解决输入序列不定长问题
            input_ids=torch.tensor(input_ids,dtype=torch.long)#转化成tensor
            #clone input_ids，然后给pad赋值-100，这样可以在后续的loss计算时忽略pad影响,也即忽略这些pad位置的loss计算
            labels=input_ids.clone()
            labels=[labels == self.tokenizer.pad_token_id] = -100
            #编写attention mask来不让pad进入attention计算
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()#非pad位置为1，pad为0
            return {
                "input_ids" : input_ids,
                "labels" : labels,
                "attention_mask" : attention_mask
            }
        
        
    