from multiprocessing import Pool
import numpy as np
import pandas as pd
import setfit

from small_text import (
    SubsamplingQueryStrategy,
    PoolBasedActiveLearner,
    RandomSampling,
    TextDataset,
)

from small_text.integrations.transformers import (
    SetFitModelArguments,
    SetFitClassificationFactory,
)
from small_text.utils.labels import list_to_csr

from sentence_transformers import losses

import query_strategy as multi_label_aal


def make_texts_and_labels(al_pool):

    print(f"Starting making texts, labels from pkl... {al_pool}")

    df_train = pd.read_pickle(al_pool)

    texts = df_train['text'].str.strip().tolist()
    nr_of_texts = len(texts)
    label_series = pd.Series([[]] * nr_of_texts).values
    labels = label_series.tolist()

    return texts, labels

def load_data(al_pool, init_data, label2int):
    print(f"Load init data from csv.. {init_data}")
    df_init = pd.read_csv(init_data, delimiter='\t', quotechar='\0')
    texts = []
    labels = []

    for index, row in df_init.iterrows():
        texts.append(row['text'].strip())
        multi_labels = [label2int[label] for label in row['labels'].split(',')]
        multi_labels = sorted(multi_labels)
        labels.append(multi_labels)

    print("Done!")

    texts_init = texts
    labels_init = labels
    num_init_samples = len(texts_init)

    with Pool(1) as pool:
        texts, labels = pool.map(make_texts_and_labels, [al_pool])[0]
    
    texts[:0] = texts_init
    labels[:0] = labels_init

    return texts, labels, num_init_samples

def get_datasets(texts, labels, num_classes, is_setfit_model):
    print("Make Textdataset...")
    nr_of_texts = len(texts)

    if labels is None:
        labels = list_to_csr([[]] * nr_of_texts, (nr_of_texts, num_classes))

    target_labels = np.arange(num_classes)

    if is_setfit_model:
        dataset = TextDataset.from_arrays(texts, labels, target_labels=target_labels)

    print("Done!")
    return dataset

def get_query_strategy(query_strategy_name, subsample_size):

    if query_strategy_name == 'random':
        query_strategy = RandomSampling()
    elif query_strategy_name == 'multi-label-aal':
        query_strategy = multi_label_aal.AdaptiveActiveLearning()
    else:
        raise ValueError(f'Unknown query strategy name: {query_strategy_name}')

    return SubsamplingQueryStrategy(query_strategy, subsample_size=subsample_size)

def get_active_learner(dataset_train, model_name, query_strategy, num_classes, is_setfit_model, log_dir, timestamp):

    if is_setfit_model:
        
        #####################
        ### Training args ###
        #####################
        max_seq_len = 512
        batch_size = 8
        seed = 13579
        sampling_strategy = 'oversampling'
        log_dir = log_dir + "__" + timestamp
        loss_fct = losses.CoSENTLoss
        use_diff_head = True

        if use_diff_head:

            print("Using differentiable head..\n")

            log_regr_strategy = 'one-vs-rest'
            train_end_to_end = True
            head_params={"out_features": num_classes, "multitarget": True}

            trainer_kwargs = {'args': setfit.TrainingArguments(
                                num_epochs=(1, 16), 
                                batch_size=(batch_size, batch_size),
                                max_length=max_seq_len,
                                sampling_strategy=sampling_strategy,
                                warmup_proportion=0.1,
                                loss=loss_fct,
                                end_to_end=train_end_to_end,
                                seed=seed,
                                logging_dir=log_dir,
                                logging_strategy='steps',
                                logging_steps=100,
                                save_strategy='epoch',
                                )}

        else:

            print("Using logistic regr head..\n")

            max_iter = 2048
            log_regr_strategy = 'one-vs-rest'
            head_params = {'max_iter': max_iter}

            trainer_kwargs = {'args': setfit.TrainingArguments(
                                num_epochs=1,
                                batch_size=batch_size,
                                max_length=max_seq_len,
                                sampling_strategy=sampling_strategy,
                                loss=loss_fct,
                                seed=seed,
                                logging_dir=log_dir,
                                logging_strategy='steps',
                                logging_steps=100,
                                save_strategy='epoch',
                                )}

        model_kwargs = {
            'multi_target_strategy': log_regr_strategy, 
            'head_params': head_params,
            }
        
        model_args = SetFitModelArguments(sentence_transformer_model=model_name, model_kwargs=model_kwargs, trainer_kwargs=trainer_kwargs)
        
        clf_factory = SetFitClassificationFactory(model_args,
                                                  num_classes,
                                                  classification_kwargs={
                                                      'device': 'cuda',
                                                      'multi_label': True,
                                                      'max_length': max_seq_len,
                                                      'use_differentiable_head': use_diff_head,
                                                      'mini_batch_size': batch_size,
                                                    })

    return PoolBasedActiveLearner(clf_factory, query_strategy, dataset_train)