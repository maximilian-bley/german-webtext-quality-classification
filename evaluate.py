from setfit import SetFitModel
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import csv
from sklearn.preprocessing import MultiLabelBinarizer


LABELS_SHORT = ['SF','GF','RF','ZF','NLC','SP']

LABELS_LONG = {
    'SF':'Sentence Boundary','GF':'Grammar Mistake','RF':'Spelling Anomaly','ZF':'Punctuation Error','NLC':'Non-linguistic Content','SP':'Letter Spacing', 'K': 'Clean',
    }

LONG2SHORT = {
    val: key
    for key, val in LABELS_LONG.items()
    }

LABEL2INT = {
    label: idx
    for idx, label in enumerate(LABELS_SHORT)
}

LABEL2INT['K'] = len(LABELS_SHORT)

INT2LABEL = {
    val: key
    for key, val in LABEL2INT.items()
}

def eval_metrics(true_labels, predicted_labels):

    precision = np.round(precision_score(true_labels, predicted_labels, average=None, zero_division=np.nan), 2)
    precision_macro = np.round(precision_score(true_labels, predicted_labels, average='macro', zero_division=np.nan), 2)
    precision_micro = np.round(precision_score(true_labels, predicted_labels, average='micro', zero_division=np.nan), 2)
    precision_sample= np.round(precision_score(true_labels, predicted_labels, average='samples', zero_division=np.nan), 2)
    
    recall = np.round(recall_score(true_labels, predicted_labels, average=None), 2)
    recall_macro = np.round(recall_score(true_labels, predicted_labels, average='macro'), 2)
    recall_micro = np.round(recall_score(true_labels, predicted_labels, average='micro'), 2)
    recall_sample = np.round(recall_score(true_labels, predicted_labels, average='samples'), 2)

    f1 = np.round(f1_score(true_labels, predicted_labels, average=None), 2)
    f1_macro = np.round(f1_score(true_labels, predicted_labels, average='macro'), 2)
    f1_micro = np.round(f1_score(true_labels, predicted_labels, average='micro'), 2)
    f1_sample = np.round(f1_score(true_labels, predicted_labels, average='samples'), 2)

    subset_accuracy = np.round(accuracy_score(true_labels, predicted_labels), 2)

    print("F1-Measures: f1, macro, micro, sample")
    print(f1, f1_macro, f1_micro, f1_sample)
    print()
    print("Precison: P, macro, micro, sample")
    print(precision ,precision_macro, precision_micro, precision_sample)
    print()
    print("Recall: R, macro, micro, sample")
    print(recall, recall_macro, recall_micro, recall_sample)
    print()
    print("Subset-Acc: ", subset_accuracy)
    print()

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="mbley/german-webtext-quality-classifier-base",
        help="Path or name of the Hugging Face model"
    )
    args = parser.parse_args()

    MODEL_PATH = args.path

    gold_df = pd.read_csv("./data/goldstandard.csv", sep="\t", quotechar='\0')
    test_set = gold_df['text'].tolist()

    def make_gold_list(label_set):
        label_set = [LONG2SHORT[label] for label in label_set.split(',')]
        return [LABEL2INT[label] for label in label_set]
    
    gold_list = gold_df['labels'].apply(make_gold_list).tolist()
    mlb = MultiLabelBinarizer(classes=list(range(len(LABEL2INT))))
    binary_array = mlb.fit_transform(gold_list)
    true_list = binary_array.tolist()

    print("Running predictions..")

    model = SetFitModel.from_pretrained(MODEL_PATH)
    pred_proba = model.predict_proba(inputs=test_set, batch_size=1, show_progress_bar=True).tolist()
    
    print(f"Labels:\n{str(list(LABELS_LONG.values()))}")
    
    for t in [0.5]:
        print("Clf-Threshold", t)
        try:
            pred_list = [
                [1 if proba > t else 0 for proba in proba_list]
                for proba_list in pred_proba]
        except TypeError:
        # for multi-output log regr
            pred_list = [
                [1 if proba_tuple[1] > t else 0 for proba_tuple in proba_list]
                for proba_list in pred_proba]

        ### adjust Clean for multi-class prediction ###
        t = 0.5
        adjusted_pred_list = []
        for proba_list in pred_proba:
            binary = [1 if proba > t else 0 for proba in proba_list]
            binary_before = [1 if proba > t else 0 for proba in proba_list]
            if any(binary[i] for i in range(len(binary)) if i != 6):
                binary[6] = 0  # class 6 is only active if no others are
            '''
            binary before [[0, 0, 0, 0, 1, 0, 1]]
            binary adjusted [[0, 0, 0, 0, 1, 0, 0]]
            '''
            adjusted_pred_list.append(binary)
        
        eval_metrics(true_list, adjusted_pred_list)

    ### save predictions ###
    label_list = [[idx for idx, value in enumerate(inner_list) if value == 1] for inner_list in adjusted_pred_list]
    label_list = [set([LABELS_LONG[INT2LABEL[idx]] for idx in inner_list]) for inner_list in label_list]
    gold_df['predicted_labels'] = label_list
    proba_list = [[round(proba, 4) for proba in inner_list] for inner_list in pred_proba]
    gold_df['predicted_probas'] = proba_list
    gold_df.to_csv('./data/goldstandard_predicted.csv', sep="\t", quoting=csv.QUOTE_NONE, quotechar="",  escapechar="\\")
