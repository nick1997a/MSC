#!/bin/bash

python name_reranker.py --retri_K 5 --acc_K 1 --trained_crossencoder '/home/gan/CODES/CrossEncoder/AAANEW/crossencoder/all_name_3v1model_bt38.pth'  2>&1 || true
python name_reranker.py --retri_K 10 --acc_K 1 --trained_crossencoder '/home/gan/CODES/CrossEncoder/AAANEW/crossencoder/all_name_3v1model_bt38.pth'  2>&1 || true
python name_reranker.py --retri_K 10 --acc_K 5 --trained_crossencoder '/home/gan/CODES/CrossEncoder/AAANEW/crossencoder/all_name_3v1model_bt38.pth'  2>&1 || true
python name_reranker.py --retri_K 15 --acc_K 5 --trained_crossencoder '/home/gan/CODES/CrossEncoder/AAANEW/crossencoder/all_name_3v1model_bt38.pth'  2>&1 || true


