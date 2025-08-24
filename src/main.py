import signal
import sys
import time
import numpy as np
import argilla.v1 as rg

from datetime import datetime
from functools import partial
from scipy.sparse import csr_matrix

from small_text.utils.labels import list_to_csr
from small_text.base import LABEL_IGNORED

from initialize import load_data, get_datasets, get_query_strategy, get_active_learner

USE_SETFIT = True
EMBEDDING_MODEL  = "./distilbert-base-multilingual-cased-sent" # could be any sentence_transformer, see: create_sentence_transformer_model.py
TIME_STAMP = datetime.now().strftime("%b%d_%H-%M-%S")

LABELS = ['SF','GF','RF','ZF','NLC','SP']

LABEL2INT = {
    label: idx
    for idx, label in enumerate(LABELS)
}

LABEL2INT['K'] = len(LABELS)

INT2LABEL = {
    val: key
    for key, val in LABEL2INT.items()
}

SUBSAMPLE_SIZE = 1024*10

def main(al_pool, query_strategy_name, output_dir, init_data, ds_name, api_url, log_dir):

    print("\nStarting AL-Experiment..")

    texts, labels, num_init_samples = load_data(al_pool, init_data, LABEL2INT)
    num_classes = len(LABEL2INT)
    ds = get_datasets(texts, list_to_csr(labels, (len(texts), num_classes)), num_classes, USE_SETFIT)
    query_strategy = get_query_strategy(query_strategy_name=query_strategy_name, subsample_size=SUBSAMPLE_SIZE)
    
    print("Using base model..", EMBEDDING_MODEL)
    active_learner = get_active_learner(ds, EMBEDDING_MODEL, query_strategy, num_classes, USE_SETFIT, log_dir, TIME_STAMP)
    active_learner.initialize(clf_or_indices=np.arange(num_init_samples))

    rg.init(api_url=api_url, api_key='argilla.apikey', workspace='argilla')
    
    argilla_dataset_name = f'{ds_name}-{query_strategy_name}'
    active_learning_loop = get_active_learning_listener(active_learner,
                                                        argilla_dataset_name,
                                                        ds,
                                                        texts,
                                                        LABEL2INT)
    
    def signal_handler(active_learner, output_path, sig, frame):
        active_learner.classifier.model.save_pretrained(output_path + "__" + TIME_STAMP)
        print('Model saved at ---> ', output_path + "__" + TIME_STAMP)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, partial(signal_handler, active_learner, output_dir))
    signal.signal(signal.SIGTERM, partial(signal_handler, active_learner, output_dir))
    
    active_learning_loop.start()


def get_active_learning_listener(active_learner, argilla_dataset_name, train_dataset, train_texts, label_dict,
                                 query_size=20):
    labels = [label_name for label_name in label_dict.keys()]
    settings = rg.TextClassificationSettings(label_schema=labels)

    # Create argilla dataset with a label schema
    rg.configure_dataset_settings(name=argilla_dataset_name, settings=settings)

    print('Querying new data points init...')
    queried_indices = active_learner.query(num_samples=query_size)
    y_pred_proba = active_learner.classifier.predict_proba(train_dataset[queried_indices])

    predictions = [
        [(INT2LABEL[i], row[i]) for i in range(y_pred_proba.shape[1]) if row[i] > 0.0]
        for row in active_learner.classifier.predict_proba(train_dataset[queried_indices])
    ]
    predictions = [lst if len(lst) > 0 else None for lst in predictions]

    print("Done!")
    
    new_batch = 0
    new_records = [
        rg.TextClassificationRecord(
            text=train_texts[idx],
            metadata={'batch_id': new_batch, 'queried_idx': idx.item(), 'log_id': i},
            id=idx.item(),
            multi_label=True,
            prediction=predictions[i],
        )
        for i, idx in enumerate(queried_indices)
    ]

    # 3. Log the batch to Argilla
    rg.log(records=new_records, name=argilla_dataset_name, num_threads=0) 

    @rg.listener(
        dataset=argilla_dataset_name,
        query="(status:Validated OR status:Discarded) AND metadata.batch_id:{batch_id}",
        condition=lambda search: search.total == query_size,
        execution_interval_in_seconds=2,
        batch_id=0,
    )
    def active_learning_loop(records, ctx):
        
        records_df = records.to_pandas().sort_values(by='event_timestamp')
        records= rg.DatasetForTextClassification.from_pandas(records_df)
        
        # 1. Update active learner
        print(f"Updating with batch_id {ctx.query_params['batch_id']} ...")

        label_list = []
        discarded = False
        for rec in records:
            if rec.status == 'Discarded':
                discarded = True
                rec.annotation = []
                rec.annotation = ['Discarded' for x in range(0, len(LABEL2INT))]

            ann_list = []
            for ann in rec.annotation:
                if ann == 'Discarded':
                    ann_list.append(LABEL_IGNORED)
                else:
                    ann_list.append(LABEL2INT[ann])
            label_list.append(ann_list)

        shape = (len(records), len(label_dict.keys()))

        if discarded:
            sparse_matrix = np.zeros(shape)
            for i, row in enumerate(label_list):
                for label in row:
                    if row[0] == -np.inf:
                        sparse_matrix[i, :len(row)] = row
                        continue
                    else:
                        sparse_matrix[i, int(label)] = 1
            y = csr_matrix(sparse_matrix)
        else:
            print("Nothing discarded!")
            y = list_to_csr([[LABEL2INT[ann] for ann in rec.annotation] for rec in records], shape)

        print(
            "Starting training! Depending on your device, this might take a while."
        )
        active_learner.update(y)
        print("Done!")

        # 2. Query active learner
        print(f"Querying new data points for batch nr {ctx.query_params['batch_id'] + 1}...")
        start_time = time.time()

        queried_indices = active_learner.query(num_samples=query_size)

        print(f"Queried idx for batch {ctx.query_params['batch_id'] + 1}")
        print(queried_indices)

        y_pred_proba = active_learner.classifier.predict_proba(train_dataset[queried_indices])
        predictions = [
            [(INT2LABEL[i], row[i]) for i in range(y_pred_proba.shape[1]) if row[i] > 0.0]
            for row in active_learner.classifier.predict_proba(train_dataset[queried_indices])
        ]
        predictions = [lst if len(lst) > 0 else None for lst in predictions]
        
        print("Done!")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Querytime: {elapsed_time:.2f} seconds")

        new_batch = ctx.query_params["batch_id"] + 1
        new_records = [
            rg.TextClassificationRecord(
                text=train_texts[idx],
                metadata={'batch_id': new_batch, 'queried_idx': idx.item(), 'log_id': i},
                id=idx.item(),
                multi_label=True,
                prediction=predictions[i],
            )
            for i, idx in enumerate(queried_indices)
        ]

        # 3. Log the batch to Argilla
        rg.log(records=new_records, name=argilla_dataset_name, num_threads=0)
        print("New data points available...")

        ctx.query_params['batch_id'] = new_batch
        print('Done!')

        print('Waiting for annotations...')

    return active_learning_loop

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--al_pool', type=str, help='input file (*.parquet)', default="")
    parser.add_argument('--query_strategy', type=str, help='query strategy', default='random')
    parser.add_argument('--output_dir', type=str, help='folder where the resulting model will be saved', default='./')
    parser.add_argument('--init_data', type=str, help='init data (*.csv)', default="")
    parser.add_argument('--ds_name', type=str, help='name of argilla dataset', default='pilotstudy')
    parser.add_argument('--api_url', type=str, help='URL to the argilla API endpoint', default='http://localhost:6900')
    parser.add_argument('--log_dir', type=str, help='dir of tensorboard logs', default='./')

    args = parser.parse_args()

    main(al_pool=args.al_pool, 
         query_strategy_name=args.query_strategy, 
         output_dir=args.output_dir, 
         init_data=args.init_data, 
         ds_name=args.ds_name,
         api_url=args.api_url,
         log_dir=args.log_dir,)