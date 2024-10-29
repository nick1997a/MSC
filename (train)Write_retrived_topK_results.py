from collections import defaultdict
import numpy as np
import json
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import faiss
from crossencoder.Models import CrossEncoder
import torch
import torch.nn as nn
import os
def load_umls(umls_dir):
    names = {}
    synonyms = defaultdict(set)
    definitions = defaultdict(list)
    semantic_types = defaultdict(set)
    broader_concepts = defaultdict(set)

    # MRCONSO.RRF contains concepts and their synonyms
    # Columns in MRCONSO.RRF explained at https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/
    with open(f'{umls_dir}/MRCONSO.RRF', encoding='utf8') as f:
        for line in f:
            split = line.rstrip('\n').split('|')
            cui = split[0]
            language = split[1]
            if language != 'ENG':
                continue

            term_type = split[12]
            entity_name = split[14]

            if cui not in names:
                names[cui] = entity_name

            # Abbreviations for term type defined at https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/abbreviations.html#mrdoc_TTY
            # These are a short list of term types for synonyms, acronyms and etc that should be exact matches to the term (and not related concepts)
            if term_type in ['PT', 'BN', 'MH', 'SY', 'SYN', 'SYGB', 'ACR', 'PN']:
                synonyms[cui].add(entity_name)

    # MRDEF.RRF contains definitions
    # Columns in MRDEF.RRF explained at https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.definitions_file_mrdef_rrf/
    # with open(f'{umls_dir}/MRDEF.RRF', encoding='utf8') as f:
    #     for line in f:
    #         split = line.rstrip('\n').split('|')
    #         cui = split[0]
    #         source = split[4]
    #         definition = split[5]
    #
    #         definitions[cui].append((source, definition))

    # MRSTY.RRF contains the semantic types of entities
    # Columns in MRSTY.RRF explained at https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.Tf/
    # with open(f'{umls_dir}/MRSTY.RRF', encoding='utf8') as f:
    #     for line in f:
    #         split = line.rstrip('\n').split('|')
    #         cui = split[0]
    #         semantic_type = split[3]
    #         semantic_types[cui].add(semantic_type)

    # # MRREL.RRF contains the relationships between entities
    # # Columns in MRREL.RRF explained at https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.related_concepts_file_mrrel_rrf/
    # with open(f'{umls_dir}/MRREL.RRF', encoding='utf8') as f:
    #     for line in f:
    #         split = line.rstrip('\n').split('|')
    #         cui1 = split[0]
    #         rel_type = split[3]
    #         cui2 = split[4]
    #
    #         # Abbreviation for relation type defined at https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/abbreviations.html#mrdoc_REL
    #         if rel_type in ['RB', 'PAR']:  # cui2 is a broader/parent concept of cui1
    #             broader_concepts[cui1].add(cui2)

    entities = []
    for cui in names:
        entity = {'cui': cui,
                  'name': names[cui],
                  'synonyms': synonyms[cui],
                  'definitions': definitions[cui],
                  'semantic_types': sorted(semantic_types[cui]),
                  'broader_concepts': [(cui2, names[cui2]) for cui2 in sorted(broader_concepts[cui]) if cui2 in names]}
        entities.append(entity)

    return entities

def generate_new_datasamples(correct_dict):
    what = 'train'
    mention_rela = []
    # with open('/home/gan/CODES/CrossEncoder/load_umls/tra_val_test/mentions/' + what + '.json', 'r',encoding='utf-8') as file1:
    #     for line in file1:
    #         mention_rela.append(json.loads(line))
    # with open('/home/gan/CODES/CrossEncoder/load_umls/tra_val_test/mentions/' + 'val' + '.json', 'r',encoding='utf-8') as file2:
    #     for line in file2:
    #         mention_rela.append(json.loads(line))
    with open('/home/gan/CODES/CrossEncoder/load_umls/tra_val_test/mentions/' + 'train' + '.json', 'r',encoding='utf-8') as file3:
        for line in file3:
            mention_rela.append(json.loads(line))

    all_pairs = []

    for i in range(0,len(mention_rela)):
        cui = mention_rela[i]['label_document_id']
        if cui in correct_dict:
            entity = correct_dict[cui]
            umls_text = entity['name']
            txt_data = {}
            txt_data['mention_text'] = mention_rela[i]['text']
            txt_data['entity_description'] = umls_text
            txt_data['label'] = 1.0
            all_pairs.append(txt_data)
    return all_pairs

def generate_umls_entities_vectors(umls_ent_names,tokenizerr,modell):
    bs = 64 # batcrh size during inference
    all_entity_embs = []
    for i in tqdm(np.arange(0, len(umls_ent_names), bs)):
        toks = tokenizerr.batch_encode_plus(umls_ent_names[i:i+bs],
                                           padding="max_length",
                                           max_length=25,
                                           truncation=True,
                                           return_tensors="pt")
        toks_cuda = {}
        for k,v in toks.items():
            toks_cuda[k] = v.cuda()
        cls_rep = modell(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
        all_entity_embs.append(cls_rep.cpu().detach().numpy())
    final_all_entity_embs = np.concatenate(all_entity_embs, axis=0)
    return final_all_entity_embs

def generate_query_vectors(alldata,tokenizerr,modell):
    bs = 64
    all_mention_info = [i['mention_text'] for i in alldata]
    all_mention_embeddings = []
    for i in tqdm(np.arange(0, len(all_mention_info), bs)):
        toks = tokenizerr.batch_encode_plus(all_mention_info[i:i+bs],
                                           padding="max_length",
                                           max_length=25,
                                           truncation=True,
                                           return_tensors="pt")
        toks_cuda = {}
        for k,v in toks.items():
            toks_cuda[k] = v.cuda()

        cls_rep = modell(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
        all_mention_embeddings.append(cls_rep.cpu().detach().numpy())
    all_mention_embeddings = np.concatenate(all_mention_embeddings, axis=0)
    return all_mention_embeddings


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

def Generate_features(tokenizer, datapath,arg_max):
    max_length = arg_max
    def tokenization(X, max_length=max_length):
        tokenized_inputs = tokenizer.encode_plus(X, padding="max_length", truncation=True, return_tensors="pt",
                                     max_length=max_length)
        return tokenized_inputs

    x_feature = list(map(lambda d: tokenization(d),datapath))
    y_feature = list(map(lambda d: 1.0,datapath))
    return x_feature,y_feature


def test(model, val_loader, loss_fn, device):
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
    return y_pred

def compute_acck( model_name, alldata,all_mention_embeddings, entities_embeddings,umls_entinames):


    query_vectors = [i for i in all_mention_embeddings]
    # query_vectors = query_vectors[:5000]
    print('length of query vectors:',len(query_vectors))
    res = faiss.StandardGpuResources()  # Use a single GPU
    index = faiss.GpuIndexFlatIP(res, 768)  # Use the GPU index
    index.add(entities_embeddings)
    accuracy = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.BCELoss()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #tokenizer.add_tokens(['[E]', '[/E]'])
    #model = CrossEncoder(model_name, 1).to(device)
    #model.load_state_dict(torch.load('/home/gan/CODES/CrossEncoder/AAANEW/crossencoder/best_model_bt52.pth'))

    result = {}

    for i in tqdm(range(0, len(query_vectors))):
        final_dict = {}
        #query_vector = query_vectors[i][np.newaxis,:]
        true_umlsname = alldata[i]['entity_description']
        k = 4
        distances, indices = index.search(query_vectors[i][np.newaxis,:], k)
        retrieved_indices = np.squeeze(indices)
        new_retrieved_indices = [int(i) for i in retrieved_indices]


        indices_names_info = [umls_entinames[i] for i in retrieved_indices]

        final_dict['terms'] = indices_names_info
        final_dict['indices'] = new_retrieved_indices

        result[i] = final_dict
        print(i)


    file_path = '(train4444)_retrieved_topK_results.json'
    with open(file_path, 'w') as f:
        json.dump(result, f)



    #
    #     if true_umlsname in indices_names_info:
    #         acc = 1
    #     else:
    #         os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #         retre_names = [umls_entinames[i] for i in retrieved_indices]
    #         true_name = alldata[i]['mention_text']
    #         rerank_pairs = [true_name+'[SEP]'+b for b in retre_names]
    #
    #
    #         x_features, y_features = Generate_features(tokenizer, rerank_pairs, 50)
    #         test_dataset = My_Dataset(x_features, y_features)
    #
    #         test_dataloader = DataLoader(test_dataset, batch_size=15, shuffle=False,
    #                                            num_workers=8)
    #
    #         # Set the model to evaluation mode (important if you are using layers like dropout, batchnorm, etc.)
    #         probs = test(model, test_dataloader,loss_fn,device)
    #
    #         result_dict = {}
    #         for i in range(0,len(probs)):
    #             result_dict[list(retrieved_indices)[i]] = probs[i]
    #         #result_dict = {i: probs[idx] for i, idx in enumerate(list(KKK))}
    #         sorted_dict = dict(sorted(result_dict.items(), key=lambda item: item[1],reverse=True))
    #
    #         ranked_indices = list(sorted_dict.keys())
    #         second_indice_info = [umls_entinames[i] for i in ranked_indices[:1]]
    #
    #         if true_umlsname in second_indice_info:
    #             acc = 1
    #         else:
    #             acc = 0
    #
    #     accuracy.append(acc)
    #
    #     if (len(accuracy)%1000==0):
    #         print(np.mean(accuracy),i)
    # print('final')
    # print(np.mean(accuracy))


def main():

    model_name = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
    path = '/home/gan/CODES/CrossEncoder/AAANEW/Exact_umls/'
    umls = load_umls(path)

    umls_entity_names = [i['name'] for i in umls]
    cui_to_entityinfo = {entity['cui']: entity for entity in umls}

    print('generate query samples')
    all_pairs = generate_new_datasamples(cui_to_entityinfo)


    tok_nizer = AutoTokenizer.from_pretrained(model_name)
    sapbert_model = AutoModel.from_pretrained(model_name).cuda()


    print('Loading umls names vectors')

    enti_vectors = np.load('Exact_umls/umls_name_vectors_sapbert.npy')
    print(enti_vectors.shape)


    print('generate query vectors')
    qu_vectors = generate_query_vectors(all_pairs,tok_nizer, sapbert_model)

    print('compute accuracy')
    compute_acck(model_name,all_pairs,qu_vectors, enti_vectors, umls_entity_names)


if __name__ == "__main__":
    main()





