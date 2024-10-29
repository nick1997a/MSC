import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (roc_curve, auc,accuracy_score,precision_recall_curve,
                             average_precision_score,f1_score,precision_score,recall_score)
from tqdm import tqdm
import pickle
from Models import CrossEncoder

import wandb

class My_Dataset(Dataset):
  def __init__(self, data, target):
    self.data = data
    self.target = target
  def __getitem__(self, idx):
    X_input_ids = self.data[idx]['input_ids']
    X_token_type_ids = self.data[idx]['token_type_ids']
    X_attention_mask = self.data[idx]['attention_mask']
    Y = self.target[idx]
    return (X_input_ids, X_token_type_ids, X_attention_mask, Y)
  def __len__(self):
    return len(self.data)

def train_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0
    y_pred, y_true = [], []
    for X_input_ids, X_token_type_ids, X_attention_mask, Y in tqdm(train_loader):
        optimizer.zero_grad()
        X_input_ids = X_input_ids.squeeze(dim=1).to(device)
        X_attention_mask = X_attention_mask.squeeze(dim=1).to(device)
        outputs = model(input_ids=X_input_ids, attention_mask=X_attention_mask)
        Y = Y.to(device).float()
        loss = loss_fn(outputs, Y)
        epoch_loss += loss
        loss.backward()
        optimizer.step()
        for item in outputs:
            y_pred.append(item.item())
        for item in Y:
            y_true.append(item.item())
    fpr, tpr, thresholds_AUC = roc_curve(y_true, y_pred)
    AUC = auc(fpr, tpr)

    AUPR = average_precision_score(y_true, y_pred)
    y_pred_list = [1 if a > 0.5 else 0 for a in y_pred]
    accuracy = accuracy_score(y_true, y_pred_list)
    F1 = f1_score(y_true, y_pred_list)
    # preci = precision_score(y_true, y_pred_list)
    # reca = recall_score(y_true, y_pred_list)
    return epoch_loss / len(train_loader), AUC, AUPR, accuracy, F1, #preci, reca


def validate(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0
    y_pred, y_true = [], []
    with torch.no_grad():
        for X_inp_id, _, X_atte_mask, Y_val in tqdm(val_loader):
            X_inp_id = X_inp_id.squeeze(dim=1).to(device)
            X_atte_mask = X_atte_mask.squeeze(dim=1).to(device)
            val_outputs = model(input_ids=X_inp_id, attention_mask=X_atte_mask)
            Y_val = Y_val.to(device).float()
            val_loss += loss_fn(val_outputs, Y_val).item()
            for item in val_outputs:
                y_pred.append(item.item())
            for item in Y_val:
                y_true.append(item.item())
        # fpr, tpr, thresholds_AUC = roc_curve(y_true, y_pred)
        # AUC = auc(fpr, tpr)
        # AUPR = average_precision_score(y_true,y_pred)
        #
        y_pred_list = [1 if a > 0.5 else 0 for a in y_pred]
        accuracy = accuracy_score(y_true,y_pred_list)
        # F1 = f1_score(y_true,y_pred_list)
        # preci = precision_score(y_true,y_pred_list)
        # reca = recall_score(y_true,y_pred_list)
    return val_loss / len(val_loader), accuracy
    # return val_loss / len(val_loader), AUC, AUPR, accuracy, F1, #preci, reca

def Generate_features(tokenizer, datapath,arg_max):
    max_length = arg_max
    def tokenization(X, max_length=max_length):
        tokenized_inputs = tokenizer.encode_plus(X, padding="max_length", truncation=True, return_tensors="pt",
                                     max_length=max_length)
        return tokenized_inputs
    json_data = []
    with open(datapath,'r') as file:
        for line in file:
            json_data.append(json.loads(line))
    x_feature = list(map(lambda d: tokenization(d['mention_text']+'[SEP]'+d['entity_description']),json_data))
    y_feature = list(map(lambda d: d['label'],json_data))
    return x_feature,y_feature

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',type=float,default=0.0003,help='learning rate')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batchsize',type=int,default=38)

    parser.add_argument('--train_dataset',type=str,
                        default='/home/gan/CODES/CrossEncoder/AAANEW/crossencoder/reformat_data/all_name/three_one.json')
    parser.add_argument('--val_dataset', type=str,
                        default='/home/gan/CODES/CrossEncoder/AAANEW/crossencoder/reformat_data/name_name/val_cro_name_name.json')

    #parser.add_argument('--pretrained_model',type=str,default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',)#default='bert-base-uncased')
    parser.add_argument('--pretrained_model', type=str,
                        default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext', )  # default='bert-base-uncased')

    parser.add_argument('--outputsize',type=int,default=1)
    parser.add_argument('--max_seqlength',type=int,default=50)
    return parser.parse_args()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using:  ", device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    tokenizer.add_tokens(['[E]','[/E]'])

    x_features, y_features = Generate_features(tokenizer, args.train_dataset,args.max_seqlength)
    train_dataset = My_Dataset(x_features, y_features)
    print('train_data length:', len(x_features))


    val_x_features, val_y_features = Generate_features(tokenizer,args.val_dataset,args.max_seqlength)
    validation_dataset = My_Dataset(val_x_features,val_y_features)
    print('validation_data length:', len(val_x_features))


    train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batchsize, shuffle=False, num_workers=8)


    model = CrossEncoder(args.pretrained_model, args.outputsize).to(device)
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)


    standard = 10000

    front = args.train_dataset.split('/')[-2].split('_')[0]
    back = args.train_dataset.split('/')[-2].split('_')[1]
    for epoch in range(0,args.epochs): #args.epochs):

        train_loss, train_AUC, train_AUPR, train_acc, train_F1 = train_epoch(model, train_dataloader, optimizer, loss_fn, device)
        print(train_loss, train_AUC, train_AUPR,train_acc, train_F1)
        #wandb.log({})  # ,'train_F1': train_F1})

        val_loss, val_acc = validate(model, validation_dataloader, loss_fn, device)
        print(f"Epoch {epoch + 1}/{args.epochs}: Train-Loss: {train_loss}  Val-Loss: {val_loss}  Val_ACC: {val_acc} ")
        if val_loss < standard:
            standard =val_loss
            torch.save(model.state_dict(), front+'_'+back+'_3v1model_' + 'bt' + str(args.batchsize) + '.pth')


if __name__ == '__main__':
    main()