import json
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
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
    pth = '/home/gan/CODES/CrossEncoder/AAANEW/crossencoder/reformat_data/all_name/train_croall_name.json'
    #with open('/home/gan/CODES/CrossEncoder/load_umls/tra_val_test/mentions/' + what + '.json', 'r',
    with open(pth, 'r',encoding='utf-8') as file1:
        for line in file1:
            mention_rela.append(json.loads(line))

    # all_pairs = []
    #
    # for i in range(0, len(mention_rela)):
    #     cui = mention_rela[i]['label_document_id']
    #     if cui in correct_dict:
    #         entity = correct_dict[cui]
    #         umls_text_name = entity['name']
    #
    #         # if len(entity['semantic_types']) == 0:
    #         #     umls_text_st = ' The ST is none. '
    #         # else:
    #         #     umls_text_st = ' The ST is '+entity['semantic_types'][0]
    #         #
    #         # if len(entity['definitions'])==0:
    #         #     umls_text_def = ' The def is none. '
    #         # else:
    #         #     umls_text_def = ' The def is '+entity['definitions'][0][1]
    #         txt_data = {}
    #         #txt_data['mention_text'] = '[E]'+mention_rela[i]['text']+'[/E]'
    #         txt_data['mention_text'] = mention_rela[i]['text']
    #         #txt_data['entity_description'] = umls_text_name + umls_text_st + umls_text_def
    #         txt_data['entity_description'] = umls_text_name
    #         txt_data['label'] = 1.0
    #         all_pairs.append(txt_data)
    jkl = []
    for i in range(0,len(mention_rela)):
        if mention_rela[i]['label']==1.0:
            jkl.append(mention_rela[i])
    return jkl

path = '/home/gan/CODES/CrossEncoder/AAANEW/Exact_umls/'

umls = load_umls(path)
umls_entity_names = [i['name'] for i in umls]
cui_to_entity = {entity['cui']: entity for entity in umls}
data = generate_new_datasamples(cui_to_entity)


print(data[:3])
file_path = '/home/gan/CODES/CrossEncoder/AAANEW/(train)_retrieved_topK_results.json'
with open(file_path,'r') as f:
    all_retrieved_topK = json.load(f)

keys = list(all_retrieved_topK.keys())
nega_pairs = []
for i in range(0,len(data)):
    if data[i]['label'] == 1.0:
        print(i+1)
        count = 0
        for each in range(0,4):
            txt_data = {}
            txt_data['mention_text'] = data[i]['mention_text']
            if count<3:
                if data[i]['entity_description'] != all_retrieved_topK[keys[i]]['terms'][each]:
                    txt_data['entity_description']=all_retrieved_topK[keys[i]]['terms'][each]
                    txt_data['label'] = 0.0
                    nega_pairs.append(txt_data)
                    count+=1


all_data = data+nega_pairs
random.seed(1)
random.shuffle(all_data)
random.shuffle(all_data)
random.shuffle(all_data)

### write all samples into a new file
count = 1
with open('three_one'+'.json','w',encoding='utf-8') as final_file:
    for i in range (0,len(all_data)):
        print(count)
        json.dump(all_data[i],final_file,separators=(',',':'))
        final_file.write('\n')
        count+=1