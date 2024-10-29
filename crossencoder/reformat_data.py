import json
import os

import numpy as np
import random
from collections import defaultdict
import argparse
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
    with open(f'{umls_dir}/MRDEF.RRF', encoding='utf8') as f:
        for line in f:
            split = line.rstrip('\n').split('|')
            cui = split[0]
            source = split[4]
            definition = split[5]

            definitions[cui].append((source, definition))

    # MRSTY.RRF contains the semantic types of entities
    # Columns in MRSTY.RRF explained at https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.Tf/
    with open(f'{umls_dir}/MRSTY.RRF', encoding='utf8') as f:
        for line in f:
            split = line.rstrip('\n').split('|')
            cui = split[0]
            semantic_type = split[3]
            semantic_types[cui].add(semantic_type)

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
def generate_train_new_datasamples(correct_dict,path,men,onto):

    document_info = []
    document_inf_dict = {}
    with open('/home/gan/CODES/CrossEncoder/load_umls/tra_val_test/documents/train.json','r',encoding='utf8') as f:
        for line in f:
            document_info.append(json.loads(line))
    for each in document_info:
        document_inf_dict[each['document_id']] = each['text']

    mention_rela = []
    with open(path, 'r',encoding='utf-8') as file1:
        for line in file1:
            mention_rela.append(json.loads(line))


    all_pairs = []
    for i in range(0, len(mention_rela)):
        cui = mention_rela[i]['label_document_id']
        if cui in correct_dict:
            entity = correct_dict[cui]

            txt_data = {}

            if men== 'name':
                txt_data['mention_text'] = mention_rela[i]['text']
            else:
                A = ' [E]'
                B = '[/E] '
                window_size = 15
                ori_document = document_inf_dict[mention_rela[i]['context_document_id']]
                ori_document = ori_document.split(' ')

                start_index = mention_rela[i]['start_index']
                end_index = mention_rela[i]['end_index']

                if start_index - window_size >= 0 and end_index + window_size <= len(ori_document):
                    txt = ' '.join(ori_document[start_index - window_size:start_index]) + A + ' '.join(
                        ori_document[start_index:end_index + 1]) + B + ' '.join(
                        ori_document[end_index + 1:end_index + window_size])
                else:
                    if start_index - window_size < 0 and end_index + window_size <= len(ori_document):
                        txt = ' '.join(ori_document[0:start_index]) + A + ' '.join(
                            ori_document[start_index:end_index + 1]) + B + ' '.join(
                            ori_document[end_index + 1:end_index + window_size])
                    else:
                        if start_index - window_size >= 0 and end_index + window_size > len(ori_document):
                            txt = ' '.join(ori_document[start_index - window_size:start_index]) + A + ' '.join(
                                ori_document[start_index:end_index + 1]) + B + ' '.join(
                                ori_document[end_index + 1: len(ori_document)])
                        else:
                            if start_index - window_size < 0 and end_index + window_size > len(ori_document):
                                txt = ' '.join(ori_document[0:start_index]) + A + ' '.join(
                                    ori_document[start_index:end_index + 1]) + B + ' '.join(
                                    ori_document[end_index + 1: len(ori_document)])
                txt_data['mention_text'] = txt


            umls_text_name = entity['name']
            if len(entity['semantic_types']) == 0:
                umls_text_st = ' The ST is none. '
            else:
                umls_text_st = ' The ST is '+entity['semantic_types'][0]
            if len(entity['definitions'])==0:
                umls_text_def = ' The def is none. '
            else:
                umls_text_def = ' The def is '+entity['definitions'][0][1]
            if onto == 'name':
                txt_data['entity_description'] = umls_text_name
            else:
                txt_data['entity_description'] = umls_text_name + umls_text_st + umls_text_def

            txt_data['label'] = 1.0
            all_pairs.append(txt_data)

    nega_pairs = []
    for i in range(0, len(all_pairs)):
        txt_data = {}
        #print('negative sample',i + 1)
        txt_data['mention_text'] = all_pairs[i]['mention_text']
        random_integer = np.random.randint(0, len(all_pairs))
        while random_integer == i:
            random_integer = np.random.randint(0, len(all_pairs))

        txt_data['entity_description'] = all_pairs[random_integer]['entity_description']
        txt_data['label'] = 0.0
        nega_pairs.append(txt_data)

    all_data = all_pairs+nega_pairs
    #all_data = data
    random.seed(1)
    random.shuffle(all_data)
    random.shuffle(all_data)
    random.shuffle(all_data)

    ### write all samples into a new file
    saved_path = './reformat_data/'+men+'_'+onto+'/'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    with open(saved_path + 'train_cro'+men+'_'+onto+'.json', 'w', encoding='utf-8') as final_file:
        for i in range(0, len(all_data)):
            json.dump(all_data[i], final_file, separators=(',', ':'))
            final_file.write('\n')
        print('The length of train data saved:', len(all_data))
    return all_pairs

def generate_val_new_datasamples(correct_dict,path,men,onto):

    document_info = []
    document_inf_dict = {}
    with open('/home/gan/CODES/CrossEncoder/load_umls/tra_val_test/documents/val.json','r',encoding='utf8') as f:
        for line in f:
            document_info.append(json.loads(line))
    for each in document_info:
        document_inf_dict[each['document_id']] = each['text']

    mention_rela = []
    with open(path, 'r',encoding='utf-8') as file1:
        for line in file1:
            mention_rela.append(json.loads(line))


    all_pairs = []
    for i in range(0, len(mention_rela)):
        cui = mention_rela[i]['label_document_id']
        if cui in correct_dict:
            entity = correct_dict[cui]

            txt_data = {}

            if men== 'name':
                txt_data['mention_text'] = mention_rela[i]['text']
            else:
                A = ' [E]'
                B = '[/E] '
                window_size = 15
                ori_document = document_inf_dict[mention_rela[i]['context_document_id']]
                ori_document = ori_document.split(' ')

                start_index = mention_rela[i]['start_index']
                end_index = mention_rela[i]['end_index']

                if start_index - window_size >= 0 and end_index + window_size <= len(ori_document):
                    txt = ' '.join(ori_document[start_index - window_size:start_index]) + A + ' '.join(
                        ori_document[start_index:end_index + 1]) + B + ' '.join(
                        ori_document[end_index + 1:end_index + window_size])
                else:
                    if start_index - window_size < 0 and end_index + window_size <= len(ori_document):
                        txt = ' '.join(ori_document[0:start_index]) + A + ' '.join(
                            ori_document[start_index:end_index + 1]) + B + ' '.join(
                            ori_document[end_index + 1:end_index + window_size])
                    else:
                        if start_index - window_size >= 0 and end_index + window_size > len(ori_document):
                            txt = ' '.join(ori_document[start_index - window_size:start_index]) + A + ' '.join(
                                ori_document[start_index:end_index + 1]) + B + ' '.join(
                                ori_document[end_index + 1: len(ori_document)])
                        else:
                            if start_index - window_size < 0 and end_index + window_size > len(ori_document):
                                txt = ' '.join(ori_document[0:start_index]) + A + ' '.join(
                                    ori_document[start_index:end_index + 1]) + B + ' '.join(
                                    ori_document[end_index + 1: len(ori_document)])
                txt_data['mention_text'] = txt


            umls_text_name = entity['name']
            if len(entity['semantic_types']) == 0:
                umls_text_st = ' The ST is none. '
            else:
                umls_text_st = ' The ST is '+entity['semantic_types'][0]
            if len(entity['definitions'])==0:
                umls_text_def = ' The def is none. '
            else:
                umls_text_def = ' The def is '+entity['definitions'][0][1]
            if onto == 'name':
                txt_data['entity_description'] = umls_text_name
            else:
                txt_data['entity_description'] = umls_text_name + umls_text_st + umls_text_def

            txt_data['label'] = 1.0
            all_pairs.append(txt_data)

    # nega_pairs = []
    # for i in range(0, len(all_pairs)):
    #     txt_data = {}
    #     print('negative sample',i + 1)
    #     txt_data['mention_text'] = all_pairs[i]['mention_text']
    #     random_integer = np.random.randint(0, len(all_pairs))
    #     while random_integer == i:
    #         random_integer = np.random.randint(0, len(all_pairs))
    #
    #     txt_data['entity_description'] = all_pairs[random_integer]['entity_description']
    #     txt_data['label'] = 0.0
    #     nega_pairs.append(txt_data)

    # all_data = all_pairs+nega_pairs
    all_data = all_pairs
    random.seed(1)
    random.shuffle(all_data)
    random.shuffle(all_data)
    random.shuffle(all_data)

    ### write all samples into a new file
    saved_path = './reformat_data/'+men+'_'+onto+'/'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    with open(saved_path + 'val_cro'+men+'_'+onto+'.json', 'w', encoding='utf-8') as final_file:
        for i in range(0, len(all_data)):
            json.dump(all_data[i], final_file, separators=(',', ':'))
            final_file.write('\n')
        print('The length of validation data saved:',len(all_data))
    return all_pairs






def parse_args():
    parser = argparse.ArgumentParser(description="A short description of your script.")

    # Add arguments here
    parser.add_argument('--umls_path', type=str, default='../Exact_umls/', help='Number of retrieval results (K)')
    parser.add_argument('--train_data_path', type=str, default='/home/gan/CODES/CrossEncoder/load_umls/tra_val_test/mentions/train.json')
    parser.add_argument('--val_data_path', type=str, default='/home/gan/CODES/CrossEncoder/load_umls/tra_val_test/mentions/val.json')
    parser.add_argument('--mention', type=str, default='all',help='Basically---name or all. name represents only mention text and all represents mention text with a window contextual info')
    parser.add_argument('--ontology', type=str, default='all',help='Basically---name or all. name represents only entity name and all represents all ontology info')

    return parser.parse_args()


def main():
    args = parse_args()
    print('umls_data_path:', args.umls_path)
    umls = load_umls(args.umls_path)
    umls_entity_names = [i['name'] for i in umls]
    cui_to_entity = {entity['cui']: entity for entity in umls}

    print('train_data_path:', args.train_data_path)
    print('val_data_path:', args.val_data_path)
    train_data = generate_train_new_datasamples(cui_to_entity,args.train_data_path,args.mention,args.ontology)
    val_data = generate_val_new_datasamples(cui_to_entity, args.val_data_path, args.mention, args.ontology)






if __name__ == "__main__":
    main()
