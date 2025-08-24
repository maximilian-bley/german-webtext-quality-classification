# german-webtext-quality-classification

Train and evaluation code for the paper **Bootstrapping a Sentence-Level Corpus Quality Classifier for Web Text using Active Learning (RANLP25)**.

Models: 
- <https://huggingface.co/mbley/german-webtext-quality-classifier-base>
- <https://huggingface.co/mbley/german-webtext-quality-classifier-small>

Data:
- <https://huggingface.co/datasets/mbley/german-webtext-quality-classification-dataset>

## Requirements

```
argilla==2.8.0
argilla-v1==1.29.1
setfit==1.1.2
```

## Installation

`pip install -r requirements.txt`

### small-text

`git clone --branch "v2.0.0.dev2" https://github.com/webis-de/small-text.git .`

Adjust `_fit` in `small-text/small_text/integrations/transformers/classifiers/setfit.py` to use customized training arguments in `main.py`

```python
def _fit(self, sub_train, sub_valid, setfit_train_kwargs):
    args = self.setfit_model_args.trainer_kwargs['args']
    trainer = setfit.Trainer(model=self.model,args=args,train_dataset=sub_train,eval_dataset=sub_valid,)
    trainer.train()
    return self
```

`pip install -e ./small-text`

## Usage

### 1. Argilla

Launch the Argilla annotation platform locally.

`docker compose -f docker-compose.yaml up -d`

### 2. Create ST

Initialize a Sentence Transformer using the base model `distilbert/distilbert-base-multilingual-cased` with mean pooling.

`src/python create_sentence_transformers_model.py`

### 3. Active Learning

Start the Active Learning loop.

```bash
python src/main.py \
    --al_pool <path_to_unlabeled_train_data> \
    --query_strategy multi-label-aal \
    --output_dir <path_to_output_dir> \
    --init_data <path_to_init_data> \
    --ds_name <argilla_dataset_name> \
    --api_url localhost:6900 \
    --log_dir <path_to_log_dir> \
```

`al_pool` needs to be a (pickled) Pandas compatible file with a column named `text`.

### 4. Evaluation

Evaluates a model on `data/goldstandard.csv`. Defaults to <https://huggingface.co/mbley/german-webtext-quality-classifier-base>.

```bash
python src/evaluate.py --path <path_to_model_dir>
```

## Acknowledgments

This work has been partially funded by the German Federal Ministry of Research, Technology, and Space (BMFTR) under the grant numbers 01IS24077A and 01IS24037B. This work has been partially supported by the ScaDS.AI Center for Scalable Data Analytics and Artificial Intelligence, project identification number: ScaDS.AI.

Computations for this work were done (in part) using resources of the Leipzig University Computing Center (<https://www.sc.uni-leipzig.de/>).
