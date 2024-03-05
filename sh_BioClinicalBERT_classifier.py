#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import random
import urllib
import pyodbc
import mysql.connector as mysql
from mysql.connector import errorcode
from mysql_connection import *
from sql_connection import *
import sys
import spacy
import medspacy
from medspacy.preprocess import Preprocessor
from medspacy.preprocess import PreprocessingRule
import re
from medspacy.ner import TargetRule
from medspacy.visualization import visualize_ent
from spacy.tokens import Span
from spacy.tokens import Doc
from target_rules import *
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as nnf
import evaluate
from torch import cuda
import sklearn
from sklearn.metrics import precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore')

# Disable parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# define data processing variables
DATA_SOURCE = 'FILE'  # define data source: database (DB) or file (FILE)
BATCH_SIZE = 400
CONCEPT_WINDOW_SIZE = 16
DATA_DIR = 'data/'
BASELINE_DIR = DATA_DIR + '/baseline'
PROMPT_DIR = DATA_DIR + '/prompt'
# specify gold standard file
GOLD_STD_FILE = "/Users/pra2008/Documents/WCMC/data/brat_suicidal_history/gold_standard_notes_ann_result.xlsx"
# specify a file to save NLP extracted output which will be used for training/testing the classifier
NLP_OUTPUT_FILE = BASELINE_DIR + "/sh_gs_data.csv"

# Setting up the device for GPU usage
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

global debug
debug = {}

# Defining some key variables that will be used later on in the classification training and validation
OUT_DIM1 = 2
MAX_LEN = 256
TRAIN_DATA_SIZE = 1000
TEST_DATA_SIZE = 1000
TRAIN_BATCH_SIZE = 16
VALIDATE_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
EPOCHS = 5
EARLY_STOP_PATIENCE = 1
EARLY_STOP_DELTA = 0.001
K_FOLD_VALIDATION = 5
LEARNING_RATE = 1e-05
DATA_SPLIT = 0.8
IS_PROMPT = False

# define a variable to hold validation results
validation_metrics = {
    "mcc1": [],
    "accuracy": [],
    "macro_precision": [],
    "micro_precision": [],
    "weighted_precision": [],
    "binary_precision": [],
    "man_precision": [],
    "macro_recall": [],
    "micro_recall": [],
    "weighted_recall": [],
    "binary_recall": [],
    "man_recall": [],
    "macro_f1": [],
    "micro_f1": [],
    "weighted_f1": [],
    "binary_f1": [],
    "man_f1": [],
    "epoch_loss": []
}

# specify classifier model Bio_ClinicalBERT
MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# load spaCy NLP pipeline
nlp = medspacy.load()
preprocessor = Preprocessor(nlp.tokenizer)
nlp.tokenizer = preprocessor


def post_in_span_between(target, modifier, span_between):
    print("Evaluating whether {0} will modify {1}".format(modifier, target))
    if "post" in span_between.text.lower():
        print("Will not modify")
        print()
        return False
    print("Will modify")
    print()
    return True


def get_target_pipeline():
    # Register extensions - is_experienced should be True by default, `is_family_history` False
    Span.set_extension("is_experienced", default=True)
    Span.set_extension("is_family_history", default=False)
    # Add rules for target entity extraction
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_matcher.add(psh_target_rules)
    return nlp


def get_suicide_history(doc):
    for ent in doc.ents:
        if ent.label_ != "SUICIDE HISTORY":
            continue
        # Check if it was family history and if it was not negated
        if ent._.is_historical and not ent._.is_negated:
            return True
    return False


def pre_process(str):
    post_str = ''
    if len(str) > 0:
        # remove non-ascii characters
        post_str = re.sub(r'[^\x00-\x7f]', r'', str)

        # remove html tags
        post_str = re.sub(r'<[^>]+>', r'', post_str)

        # remove punctuations and numbers
        post_str = re.sub('[^a-zA-Z]', ' ', post_str)

        # remove single character
        post_str = re.sub(r"s+[a-zA-Z]s+", ' ', post_str)

        # remove multiple spaces
        post_str = re.sub(r'\s+', ' ', post_str)

    return post_str


def get_file_count(file_name):
    count = 0
    notes_df = pd.read_excel(file_name)
    count = notes_df.shape[0]
    return count


def get_file_data(file_name, size, start):
    documents = []
    notes_df = pd.read_excel(file_name)
    end = start + size
    batch_df = notes_df[start:end]
    batch_df = batch_df.reset_index()  # make sure indexes pair with number of rows
    for index, row in batch_df.iterrows():
        document = (row['id'], row['note_text'], row['sh_label'])
        documents.append(document)
    return documents


def process_file_data():
    print('Preparing NLP pipelines...')
    target_pipeline = get_target_pipeline()
    total = get_file_count(GOLD_STD_FILE)
    results = []
    if total:
        for offset in range(0, total, BATCH_SIZE):
            print('Fetching {0} documents; offset {1} '.format(BATCH_SIZE, offset))
            records = get_file_data(GOLD_STD_FILE, BATCH_SIZE, offset)
            # process documents through the pipeline
            if records:
                total = len(records)
                for record in records:
                    # pre-process text to remove non-printable characters.
                    doc_id = record[0]
                    text = pre_process(record[1])
                    doc_label = record[2]
                    # text = """Yes.  · SI in the past, no SA."""
                    if doc_id and text:
                        doc = target_pipeline(text)
                        note_length = len(doc.text)
                        # visualize_ent(doc)
                        # print(doc._.context_graph.modifiers)
                        snippets = ""
                        if len(doc.ents) > 0:
                            instance = 0
                            for ent in doc.ents:
                                instance = instance + 1
                                # get snippets of this entity
                                right_token = ent.end + CONCEPT_WINDOW_SIZE
                                left_token = ent.start - CONCEPT_WINDOW_SIZE
                                if left_token < 0:
                                    left_token = ent.start
                                snippet = doc[left_token:right_token]
                                snippets = snippets + " " + str(snippet.text)
                        else:
                            left_token = random.randrange(note_length)
                            right_token = left_token + 6 * CONCEPT_WINDOW_SIZE
                            if right_token > note_length:
                                right_token = note_length
                            snippet = text[left_token:right_token]
                            snippets = snippet
                        item = (record[0], snippets, doc_label)
                        results.append(item)
                print('Completed {} documents; offset {}'.format(BATCH_SIZE, offset))
    return results


# split data into test and train
def data_split_train_test(df, train_test_split=0.8, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_test_split * m)
    df_train = df.iloc[perm[:train_end]]
    df_test = df.iloc[perm[train_end:]]
    return df_train, df_test


def create_baseline_data(df):
    # Create baseline data
    suicide_id = []
    suicide_input = []
    suicide_label = []
    for i in range(df.shape[0]):
        suicide_id.append(df.iloc[i]['id'])
        input_sent = df.iloc[i]['SUICIDE_HISTORY']
        input_sent = re.sub(r"[\n\r\t]", ' ', input_sent)
        input_sent = re.sub(r"[\"]", '', input_sent)
        suicide_input.append(input_sent)
        suicide_label.append(df.iloc[i]['label'])
    df_suicide = pd.DataFrame({'id': suicide_id, 'input': suicide_input, 'label': suicide_label})
    return df_suicide


def create_prompt_data(df):
    suicide_id = []
    suicide_input = []
    suicide_label = []
    for i in range(df.shape[0]):
        suicide_id.append(df.iloc[i]['id'])
        if isinstance(df.iloc[i]['SUICIDE_HISTORY'], float):
            suicide_input_sent = str(df.iloc[i]['SUICIDE_HISTORY'])
        else:
            suicide_input_sent = df.iloc[i]['SUICIDE_HISTORY']
            suicide_input_sent = re.sub(r"[\n\r\t]", ' ', suicide_input_sent)
            suicide_input_sent = re.sub(r"[\"]", '', suicide_input_sent)
        if df.iloc[i]['label'] == 0:
            suicide_input_sent = suicide_input_sent + " [SEP] Family suicide history: No passages about patient’s family suicide history."
        else:
            suicide_input_sent = suicide_input_sent + " [SEP] Family suicide history: Passages about patient’s family suicide history."
        suicide_input.append(suicide_input_sent)
        suicide_label.append(df.iloc[i]['label'])
    df_suicide = pd.DataFrame({'id': suicide_id, 'input': suicide_input, 'label': suicide_label})
    return df_suicide


def compute_man_metrics(ref, prd):
    man_precision = 0.0
    man_recall = 0.0
    man_f1_score = 0.0
    ref_df = pd.DataFrame(ref.numpy())
    prd_df = pd.DataFrame(prd.numpy())
    join_df = pd.concat([ref_df, prd_df], axis=1, join='inner')
    join_df.columns = ['ref', 'prd']
    tp_count = len(join_df.loc[(join_df['ref'] == 1) & (join_df['prd'] == 1)])
    tn_count = len(join_df.loc[(join_df['ref'] == 0) & (join_df['prd'] == 0)])
    fp_count = len(join_df.loc[(join_df['ref'] == 0) & (join_df['prd'] == 1)])
    fn_count = len(join_df.loc[(join_df['ref'] == 1) & (join_df['prd'] == 0)])
    # print(f"TP:{tp_count}, TN = {tn_count}, FP = {fp_count}, FN={fn_count}")
    if tp_count == 0 & fp_count == 0:
        man_precision = 0.0
    else:
        man_precision = tp_count / (tp_count + fp_count)
    if tp_count == 0 & fn_count == 0:
        man_recall = 0.0
    else:
        man_recall = tp_count / (tp_count + fn_count)

    if man_precision > 0.0 and man_recall > 0.0:
        man_f1_score = (2 * man_precision * man_recall) / (man_precision + man_recall)
    else:
        man_f1_score = 0.0

    # print(f'Precision: {man_precision}, Recall: {man_recall}, F1 score: {man_f1_score}')
    return man_precision, man_recall, man_f1_score


def save_df_data(df_del, save_file):
    df_del.to_csv(save_file)


def read_df_data(data_file):
    data_df = pd.read_csv(data_file)
    return data_df


class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        title = str(self.data.input[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'target1': torch.tensor(self.data.label[index], dtype=torch.long)
        }

    def __len__(self):
        return self.len


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.
class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(MODEL_NAME)

        self.pre_classifier1 = torch.nn.Linear(768, 768)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.classifier1 = torch.nn.Linear(768, OUT_DIM1)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]

        pooler1 = hidden_state[:, 0]
        pooler1 = self.pre_classifier1(pooler1)
        pooler1 = torch.nn.ReLU()(pooler1)
        pooler1 = self.dropout1(pooler1)
        output1 = self.classifier1(pooler1)

        return output1


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def test(model, testing_loader, loss_function):
    tr_loss = 0
    mcc1 = 0
    accuracy = 0
    macro_precision = 0
    micro_precision = 0
    weighted_precision = 0
    binary_precision = 0
    man_precision = 0
    macro_recall = 0
    micro_recall = 0
    weighted_recall = 0
    binary_recall = 0
    man_recall = 0
    macro_f1_1 = 0
    micro_f1_1 = 0
    weighted_f1_1 = 0
    binary_f1_1 = 0
    man_f1_1 = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            target1 = data['target1'].to(device, dtype=torch.long)
            output1 = model(ids, mask)

            loss1 = loss_function(output1, target1)
            loss = loss1
            tr_loss += loss.item()

            big_val1, big_idx1 = torch.max(output1.data, dim=1)

            matthews_metric = evaluate.load("matthews_correlation")
            mcc_result1 = matthews_metric.compute(references=target1,
                                                  predictions=big_idx1)['matthews_correlation']

            accuracy_metric = evaluate.load("accuracy")
            accuracy_result1 = accuracy_metric.compute(predictions=big_idx1, references=target1)['accuracy']

            man_precision_result1, man_recall_result1, man_f1_result1 = compute_man_metrics(target1, big_idx1)

            precision_metric = evaluate.load("precision")
            macro_precision_result1 = precision_metric.compute(predictions=big_idx1, references=target1, average='macro')['precision']
            micro_precision_result1 = precision_metric.compute(predictions=big_idx1, references=target1, average='micro')['precision']
            weighted_precision_result1 = precision_metric.compute(predictions=big_idx1, references=target1, average='weighted')['precision']
            binary_precision_result1 = precision_metric.compute(predictions=big_idx1, references=target1, average='binary')['precision']

            recall_metric = evaluate.load("recall")
            macro_recall_result1 = recall_metric.compute(predictions=big_idx1, references=target1, average='macro')['recall']
            micro_recall_result1 = recall_metric.compute(predictions=big_idx1, references=target1, average='micro')['recall']
            weighted_recall_result1 = recall_metric.compute(predictions=big_idx1, references=target1, average='weighted')['recall']
            binary_recall_result1 = recall_metric.compute(predictions=big_idx1, references=target1, average='binary')['recall']

            f1_metric = evaluate.load("f1")
            macro_f1_result1 = f1_metric.compute(predictions=big_idx1, references=target1, average='macro')['f1']
            micro_f1_result1 = f1_metric.compute(predictions=big_idx1, references=target1, average='micro')['f1']
            weighted_f1_result1 = f1_metric.compute(predictions=big_idx1, references=target1, average='weighted')['f1']
            binary_f1_result1 = f1_metric.compute(predictions=big_idx1, references=target1, average='binary')['f1']

            mcc1 += mcc_result1
            accuracy += accuracy_result1
            macro_precision += macro_precision_result1
            micro_precision += micro_precision_result1
            weighted_precision += weighted_precision_result1
            binary_precision += binary_precision_result1
            man_precision += man_precision_result1

            macro_recall += macro_recall_result1
            micro_recall += micro_recall_result1
            weighted_recall += weighted_recall_result1
            binary_recall += binary_recall_result1
            man_recall += man_recall_result1

            macro_f1_1 += macro_f1_result1
            micro_f1_1 += micro_f1_result1
            weighted_f1_1 += weighted_f1_result1
            binary_f1_1 += binary_f1_result1
            man_f1_1 += man_f1_result1

            nb_tr_steps += 1
            nb_tr_examples += target1.size(0)

    print('----------------------- BEGIN TEST RESULTS -------------------------------')
    print("task1: presence")

    print(f'The Total MCC for Test: {(mcc1 * 100) / nb_tr_steps}')

    print(f'The Total accuracy for Test: {(accuracy * 100) / nb_tr_steps}')

    print(f'The Total macro_precision for Test: {(macro_precision * 100) / nb_tr_steps}')
    print(f'The Total micro_precision for Test: {(micro_precision * 100) / nb_tr_steps}')
    print(f'The Total weighted_precision for Test: {(weighted_precision * 100) / nb_tr_steps}')
    print(f'The Total binary_precision for Test: {(binary_precision * 100) / nb_tr_steps}')
    print(f'The Total man_precision for Test: {(man_precision * 100) / nb_tr_steps}')

    print(f'The Total macro_recall for Test: {(macro_recall * 100) / nb_tr_steps}')
    print(f'The Total micro_recall for Test: {(micro_recall * 100) / nb_tr_steps}')
    print(f'The Total weighted_recall for Test: {(weighted_recall * 100) / nb_tr_steps}')
    print(f'The Total binary_recall for Test: {(binary_recall * 100) / nb_tr_steps}')
    print(f'The Total man_recall for Test: {(man_recall * 100) / nb_tr_steps}')

    print(f'The Total macro_f1 for Test: {(macro_f1_1 * 100) / nb_tr_steps}')
    print(f'The Total micro_f1 for Test: {(micro_f1_1 * 100) / nb_tr_steps}')
    print(f'The Total weighted_f1 for Test: {(weighted_f1_1 * 100) / nb_tr_steps}')
    print(f'The Total binary_f1 for Test: {(binary_f1_1 * 100) / nb_tr_steps}')
    print(f'The Total man_f1 for Test: {(man_f1_1 * 100) / nb_tr_steps}')

    epoch_loss = tr_loss / nb_tr_steps
    print(f"Validation Loss Epoch: {epoch_loss}")
    print('-------------------------- END TEST RESULTS --------------------------------')
    return


def validate(epoch, model, validate_loader, loss_function):
    model.eval()
    tr_loss = 0
    mcc1 = 0
    accuracy = 0
    macro_precision = 0
    micro_precision = 0
    weighted_precision = 0
    binary_precision = 0
    man_precision = 0
    macro_recall = 0
    micro_recall = 0
    weighted_recall = 0
    binary_recall = 0
    man_recall = 0
    macro_f1_1 = 0
    micro_f1_1 = 0
    weighted_f1_1 = 0
    binary_f1_1 = 0
    man_f1_1 = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    with torch.no_grad():
        for _, data in enumerate(validate_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            target1 = data['target1'].to(device, dtype=torch.long)
            output1 = model(ids, mask)

            loss1 = loss_function(output1, target1)
            loss = loss1
            tr_loss += loss.item()

            big_val1, big_idx1 = torch.max(output1.data, dim=1)

            matthews_metric = evaluate.load("matthews_correlation")
            mcc_result1 = matthews_metric.compute(references=target1,
                                                  predictions=big_idx1)['matthews_correlation']

            accuracy_metric = evaluate.load("accuracy")
            accuracy_result1 = accuracy_metric.compute(predictions=big_idx1, references=target1)['accuracy']

            man_precision_result1, man_recall_result1, man_f1_result1 = compute_man_metrics(target1, big_idx1)

            precision_metric = evaluate.load("precision")
            macro_precision_result1 = precision_metric.compute(predictions=big_idx1, references=target1, average='macro')['precision']
            micro_precision_result1 = precision_metric.compute(predictions=big_idx1, references=target1, average='micro')['precision']
            weighted_precision_result1 = precision_metric.compute(predictions=big_idx1, references=target1, average='weighted')['precision']
            binary_precision_result1 = precision_metric.compute(predictions=big_idx1, references=target1, average='binary')['precision']

            recall_metric = evaluate.load("recall")
            macro_recall_result1 = recall_metric.compute(predictions=big_idx1, references=target1, average='macro')['recall']
            micro_recall_result1 = recall_metric.compute(predictions=big_idx1, references=target1, average='micro')['recall']
            weighted_recall_result1 = recall_metric.compute(predictions=big_idx1, references=target1, average='weighted')['recall']
            binary_recall_result1 = recall_metric.compute(predictions=big_idx1, references=target1, average='binary')['recall']

            f1_metric = evaluate.load("f1")
            macro_f1_result1 = f1_metric.compute(predictions=big_idx1, references=target1, average='macro')['f1']
            micro_f1_result1 = f1_metric.compute(predictions=big_idx1, references=target1, average='micro')['f1']
            weighted_f1_result1 = f1_metric.compute(predictions=big_idx1, references=target1, average='weighted')['f1']
            binary_f1_result1 = f1_metric.compute(predictions=big_idx1, references=target1, average='binary')['f1']

            mcc1 += mcc_result1

            accuracy += accuracy_result1

            macro_precision += macro_precision_result1
            micro_precision += micro_precision_result1
            weighted_precision += weighted_precision_result1
            binary_precision += binary_precision_result1
            man_precision += man_precision_result1

            macro_recall += macro_recall_result1
            micro_recall += micro_recall_result1
            weighted_recall += weighted_recall_result1
            binary_recall += binary_recall_result1
            man_recall += man_recall_result1

            macro_f1_1 += macro_f1_result1
            micro_f1_1 += micro_f1_result1
            weighted_f1_1 += weighted_f1_result1
            binary_f1_1 += binary_f1_result1
            man_f1_1 += man_f1_result1

            nb_tr_steps += 1
            nb_tr_examples += target1.size(0)

    print("task1: presence")
    val_mcc1 = (mcc1 * 100) / nb_tr_steps
    print(f'The Total MCC for Epoch {epoch}: {val_mcc1}')
    val_accuracy = (accuracy * 100) / nb_tr_steps
    print(f'The Total accuracy for Epoch {epoch}: {val_accuracy}')
    val_macro_precision = (macro_precision * 100) / nb_tr_steps
    print(f'The Total macro_precision for Epoch {epoch}: {val_macro_precision}')
    val_micro_precision = (micro_precision * 100) / nb_tr_steps
    print(f'The Total micro_precision for Epoch {epoch}: {val_micro_precision}')
    val_weighted_precision = (weighted_precision * 100) / nb_tr_steps
    print(f'The Total weighted_precision for Epoch {epoch}: {val_weighted_precision}')
    val_binary_precision = (binary_precision * 100) / nb_tr_steps
    print(f'The Total binary_precision for Epoch {epoch}: {val_binary_precision}')
    val_man_precision = (man_precision * 100) / nb_tr_steps
    print(f'The Total man_precision for Epoch {epoch}: {val_man_precision}')

    val_macro_recall = (macro_recall * 100) / nb_tr_steps
    print(f'The Total macro_recall for Epoch {epoch}: {val_macro_recall}')
    val_micro_recall = (micro_recall * 100) / nb_tr_steps
    print(f'The Total micro_recall for Epoch {epoch}: {val_micro_recall}')
    val_weighted_recall = (weighted_recall * 100) / nb_tr_steps
    print(f'The Total weighted_recall for Epoch {epoch}: {val_weighted_recall}')
    val_binary_recall = (binary_recall * 100) / nb_tr_steps
    print(f'The Total binary_recall for Epoch {epoch}: {val_binary_recall}')
    val_man_recall = (man_recall * 100) / nb_tr_steps
    print(f'The Total man_recall for Epoch {epoch}: {val_man_recall}')

    val_macro_f1 = (macro_f1_1 * 100) / nb_tr_steps
    print(f'The Total macro_f1 for Epoch {epoch}: {val_macro_f1}')
    val_micro_f1 = (micro_f1_1 * 100) / nb_tr_steps
    print(f'The Total micro_f1 for Epoch {epoch}: {val_micro_f1}')
    val_weighted_f1 = (weighted_f1_1 * 100) / nb_tr_steps
    print(f'The Total weighted_f1 for Epoch {epoch}: {val_weighted_f1}')
    val_binary_f1 = (binary_f1_1 * 100) / nb_tr_steps
    print(f'The Total binary_f1 for Epoch {epoch}: {val_binary_f1}')
    val_man_f1 = (man_f1_1 * 100) / nb_tr_steps
    print(f'The Total man_f1 for Epoch {epoch}: {val_man_f1}')

    epoch_loss = tr_loss / nb_tr_steps
    print(f"Validation Loss Epoch: {epoch_loss}")

    validation_metrics["mcc1"] += [val_mcc1]
    validation_metrics["accuracy"] += [val_accuracy]
    validation_metrics["macro_precision"] += [val_macro_precision]
    validation_metrics["micro_precision"] += [val_micro_precision]
    validation_metrics["weighted_precision"] += [val_weighted_precision]
    validation_metrics["binary_precision"] += [val_binary_precision]
    validation_metrics["man_precision"] += [val_man_precision]
    validation_metrics["macro_recall"] += [val_macro_recall]
    validation_metrics["micro_recall"] += [val_micro_recall]
    validation_metrics["weighted_recall"] += [val_weighted_recall]
    validation_metrics["binary_recall"] += [val_binary_recall]
    validation_metrics["man_recall"] += [val_man_recall]
    validation_metrics["macro_f1"] += [val_macro_f1]
    validation_metrics["micro_f1"] += [val_micro_f1]
    validation_metrics["weighted_f1"] += [val_weighted_f1]
    validation_metrics["binary_f1"] += [val_binary_f1]
    validation_metrics["man_f1"] += [val_man_f1]
    validation_metrics["epoch_loss"] += [epoch_loss]

    return


# Defining the training function on the 80% of the data set for tuning the distilbert model
def train(epoch, model, loss_function, optimizer, training_loader, validate_loader):
    global debug
    tr_loss = 0
    mcc1 = 0
    accuracy = 0
    macro_precision = 0
    micro_precision = 0
    weighted_precision = 0
    binary_precision = 0
    man_precision = 0
    macro_recall = 0
    micro_recall = 0
    weighted_recall = 0
    binary_recall = 0
    man_recall = 0
    macro_f1_1 = 0
    micro_f1_1 = 0
    weighted_f1_1 = 0
    binary_f1_1 = 0
    man_f1_1 = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        target1 = data['target1'].to(device, dtype=torch.long)
        output1 = model(ids, mask)

        debug['target1'] = target1
        debug['output1'] = output1

        loss1 = loss_function(output1, target1)
        loss = loss1
        tr_loss += loss.item()

        big_val1, big_idx1 = torch.max(output1.data, dim=1)

        matthews_metric = evaluate.load("matthews_correlation")
        mcc_result1 = matthews_metric.compute(references=target1,
                                              predictions=big_idx1)['matthews_correlation']

        accuracy_metric = evaluate.load("accuracy")
        accuracy_result1 = accuracy_metric.compute(predictions=big_idx1, references=target1)['accuracy']

        man_precision_result1, man_recall_result1, man_f1_result1 = compute_man_metrics(target1, big_idx1)

        precision_metric = evaluate.load("precision")
        macro_precision_result1 = precision_metric.compute(predictions=big_idx1, references=target1, average='macro')['precision']
        micro_precision_result1 = precision_metric.compute(predictions=big_idx1, references=target1, average='micro')['precision']
        weighted_precision_result1 = precision_metric.compute(predictions=big_idx1, references=target1, average='weighted')['precision']
        binary_precision_result1 = precision_metric.compute(predictions=big_idx1, references=target1, average='binary')['precision']

        recall_metric = evaluate.load("recall")
        macro_recall_result1 = recall_metric.compute(predictions=big_idx1, references=target1, average='macro')['recall']
        micro_recall_result1 = recall_metric.compute(predictions=big_idx1, references=target1, average='micro')['recall']
        weighted_recall_result1 = recall_metric.compute(predictions=big_idx1, references=target1, average='weighted')['recall']
        binary_recall_result1 = recall_metric.compute(predictions=big_idx1, references=target1, average='binary')['recall']

        f1_metric = evaluate.load("f1")
        macro_f1_result1 = f1_metric.compute(predictions=big_idx1, references=target1, average='macro')['f1']
        micro_f1_result1 = f1_metric.compute(predictions=big_idx1, references=target1, average='micro')['f1']
        weighted_f1_result1 = f1_metric.compute(predictions=big_idx1, references=target1, average='weighted')['f1']
        binary_f1_result1 = f1_metric.compute(predictions=big_idx1, references=target1, average='binary')['f1']

        mcc1 += mcc_result1

        accuracy += accuracy_result1

        macro_precision += macro_precision_result1
        micro_precision += micro_precision_result1
        weighted_precision += weighted_precision_result1
        binary_precision += binary_precision_result1
        man_precision += man_precision_result1

        macro_recall += macro_recall_result1
        micro_recall += micro_recall_result1
        weighted_recall += weighted_recall_result1
        binary_recall += binary_recall_result1
        man_recall += man_recall_result1

        macro_f1_1 += macro_f1_result1
        micro_f1_1 += micro_f1_result1
        weighted_f1_1 += weighted_f1_result1
        binary_f1_1 += binary_f1_result1
        man_f1_1 += man_f1_result1

        nb_tr_steps += 1
        nb_tr_examples += target1.size(0)

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print("task1: presence")

    print(f'The Total MCC for Epoch {epoch}: {(mcc1 * 100) / nb_tr_steps}')

    print(f'The Total accuracy for Epoch {epoch}: {(accuracy * 100) / nb_tr_steps}')

    print(f'The Total macro_precision for Epoch {epoch}: {(macro_precision * 100) / nb_tr_steps}')
    print(f'The Total micro_precision for Epoch {epoch}: {(micro_precision * 100) / nb_tr_steps}')
    print(f'The Total weighted_precision for Epoch {epoch}: {(weighted_precision * 100) / nb_tr_steps}')
    print(f'The Total binary_precision for Epoch {epoch}: {(binary_precision * 100) / nb_tr_steps}')
    print(f'The Total man_precision for Epoch {epoch}: {(man_precision * 100) / nb_tr_steps}')

    print(f'The Total macro_recall for Epoch {epoch}: {(macro_recall * 100) / nb_tr_steps}')
    print(f'The Total micro_recall for Epoch {epoch}: {(micro_recall * 100) / nb_tr_steps}')
    print(f'The Total weighted_recall for Epoch {epoch}: {(weighted_recall * 100) / nb_tr_steps}')
    print(f'The Total binary_recall for Epoch {epoch}: {(binary_recall * 100) / nb_tr_steps}')
    print(f'The Total man_recall for Epoch {epoch}: {(man_recall * 100) / nb_tr_steps}')

    print(f'The Total macro_f1 for Epoch {epoch}: {(macro_f1_1 * 100) / nb_tr_steps}')
    print(f'The Total micro_f1 for Epoch {epoch}: {(micro_f1_1 * 100) / nb_tr_steps}')
    print(f'The Total weighted_f1 for Epoch {epoch}: {(weighted_f1_1 * 100) / nb_tr_steps}')
    print(f'The Total binary_f1 for Epoch {epoch}: {(binary_f1_1 * 100) / nb_tr_steps}')
    print(f'The Total man_f1 for Epoch {epoch}: {(man_f1_1 * 100) / nb_tr_steps}')

    epoch_loss = tr_loss / nb_tr_steps
    print(f"Training Loss Epoch: {epoch_loss}")

    print('This is the validation section to print the accuracy and see how it performs')
    print('Here we are leveraging on the dataloader crearted for the validation data set, the approcah is using more of pytorch')

    validate(epoch, model, validate_loader, loss_function)
    print('------------------------------------------------------------------')

    return


def run_classifier(training_data, testing_data):
    # Model building
    model = ModelClass()
    model.to(device)

    # Creating the loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    early_stopper = EarlyStopper(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_DELTA)

    # train and cross-validate model
    for epoch in range(EPOCHS):
        # Reset validation metrics
        # for value in validation_metrics.values():
        #     del value[:]
        validation_metrics.update((k, []) for k in validation_metrics)

        # train(epoch, model, loss_function, optimizer, training_loader, testing_loader)
        total_size = len(training_data)
        fraction = 1 / K_FOLD_VALIDATION
        seg = int(total_size * fraction)
        for i in range(K_FOLD_VALIDATION):
            train_ll = 0
            train_lr = i * seg
            validate_l = train_lr
            validate_r = i * seg + seg
            train_rl = validate_r
            train_rr = total_size
            train_left_indices = list(range(train_ll, train_lr))
            train_right_indices = list(range(train_rl, train_rr))
            train_indices = train_left_indices + train_right_indices
            validate_indices = list(range(validate_l, validate_r))

            train_set = training_data.iloc[train_indices]
            train_set = train_set.reset_index()  # make sure indexes pair with number of rows
            validate_set = training_data.iloc[validate_indices]
            validate_set = validate_set.reset_index()  # make sure indexes pair with number of rows

            # Data preparation
            training_set = Triage(train_set, tokenizer, MAX_LEN)
            validate_set = Triage(validate_set, tokenizer, MAX_LEN)
            train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
            validate_params = {'batch_size': VALIDATE_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
            training_loader = DataLoader(training_set, **train_params)
            validate_loader = DataLoader(validate_set, **validate_params)
            # train-validate
            train(epoch, model, loss_function, optimizer, training_loader, validate_loader)

        # print grand validation results for this epoch
        validation_results = {k: sum(v) / len(v) for k, v in validation_metrics.items()}
        print(f'------------BEGIN VALIDATION RESULTS FOR EPOCH:{epoch} --------------')
        print(f'The Total MCC for Epoch {epoch}: {validation_results["mcc1"]}')
        print(f'The Total accuracy for Epoch {epoch}: {validation_results["accuracy"]}')
        print(f'The Total macro_precision for Epoch {epoch}: {validation_results["macro_precision"]}')
        print(f'The Total micro_precision for Epoch {epoch}: {validation_results["micro_precision"]}')
        print(f'The Total weighted_precision for Epoch {epoch}: {validation_results["weighted_precision"]}')
        print(f'The Total binary_precision for Epoch {epoch}: {validation_results["binary_precision"]}')
        print(f'The Total man_precision for Epoch {epoch}: {validation_results["man_precision"]}')
        print(f'The Total macro_recall for Epoch {epoch}: {validation_results["macro_recall"]}')
        print(f'The Total micro_recall for Epoch {epoch}: {validation_results["micro_recall"]}')
        print(f'The Total weighted_recall for Epoch {epoch}: {validation_results["weighted_recall"]}')
        print(f'The Total binary_recall for Epoch {epoch}: {validation_results["binary_recall"]}')
        print(f'The Total man_recall for Epoch {epoch}: {validation_results["man_recall"]}')
        print(f'The Total macro_f1 for Epoch {epoch}: {validation_results["macro_f1"]}')
        print(f'The Total micro_f1 for Epoch {epoch}: {validation_results["micro_f1"]}')
        print(f'The Total weighted_f1 for Epoch {epoch}: {validation_results["weighted_f1"]}')
        print(f'The Total binary_f1 for Epoch {epoch}: {validation_results["binary_f1"]}')
        print(f'The Total man_f1 for Epoch {epoch}: {validation_results["man_f1"]}')
        print(f'The Total Loss for Epoch {epoch}: {validation_results["epoch_loss"]}')
        print(f'------------END VALIDATION RESULTS FOR EPOCH:{epoch} --------------')

        # stop, if current validation loss not improving
        current_epoch_loss = validation_results["epoch_loss"]
        if early_stopper.early_stop(current_epoch_loss):
            break

    # finally test the model
    testing_set = Triage(testing_data, tokenizer, MAX_LEN)
    test_params = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    testing_loader = DataLoader(testing_set, **test_params)
    test(model, testing_loader, loss_function)


if __name__ == '__main__':
    rand_seed = random.randint(0, 100)
    if DATA_SOURCE == 'FILE':
        # extract concept specific snippets from documents through NLP
        print('Begin data extraction by NLP')
        extracted_data = process_file_data()
        print('Completed data extraction by NLP')
        if len(extracted_data) > 0:
            # pre-process, create train/test data
            print('Begin NLP data pre processing')
            df = pd.DataFrame(extracted_data, columns=['id', 'SUICIDE_HISTORY', 'label'], dtype=float)
            # save the data frame for future run
            save_df_data(df, NLP_OUTPUT_FILE)
            print('Completed NLP data processing. Output saved to {}'.format(NLP_OUTPUT_FILE))

            print('Begin modelling classifier')
            # split data into train and test
            train_df, test_df = data_split_train_test(df, DATA_SPLIT, rand_seed)
            if IS_PROMPT:
                # create prompt data
                train_prompt_data = create_prompt_data(train_df)
                test_prompt_data = create_prompt_data(test_df)
                train_data = train_prompt_data.iloc[:TRAIN_DATA_SIZE]
                test_data = test_prompt_data.iloc[:TEST_DATA_SIZE]
            else:
                # create baseline data
                train_baseline_data = create_baseline_data(train_df)
                test_baseline_data = create_baseline_data(test_df)
                train_data = train_baseline_data.iloc[:TRAIN_DATA_SIZE]
                test_data = test_baseline_data.iloc[:TEST_DATA_SIZE]

            print("TRAIN data set: {}".format(train_data.shape))
            print("TEST data set: {}".format(test_data.shape))

            # run classifier
            print('Begin model building and training')
            run_classifier(train_data, test_data)
            print('Completed model building, training and validation')
        else:
            print('Found no records to process')
    elif DATA_SOURCE == 'TRAIN_TEST_DF':
        print('Reading data from file: {}'.format(NLP_OUTPUT_FILE))
        # read data from previously saved file
        df = read_df_data(NLP_OUTPUT_FILE)
        print('Begin modelling classifier')
        # change column names
        train_df, test_df = data_split_train_test(df, DATA_SPLIT, rand_seed)
        if IS_PROMPT:
            # create prompt data
            train_prompt_data = create_prompt_data(train_df)
            test_prompt_data = create_prompt_data(test_df)
            train_data = train_prompt_data.iloc[:TRAIN_DATA_SIZE]
            test_data = test_prompt_data.iloc[:TEST_DATA_SIZE]
        else:
            # create baseline data
            train_baseline_data = create_baseline_data(train_df)
            test_baseline_data = create_baseline_data(test_df)
            train_data = train_baseline_data.iloc[:TRAIN_DATA_SIZE]
            test_data = test_baseline_data.iloc[:TEST_DATA_SIZE]

        print("TRAIN data set: {}".format(train_data.shape))
        print("TEST data set: {}".format(test_data.shape))

        # run classifier
        print('Begin model building and training')
        run_classifier(train_data, test_data)
        print('Completed model building, training and validation')

    print('Completed running classifier')
