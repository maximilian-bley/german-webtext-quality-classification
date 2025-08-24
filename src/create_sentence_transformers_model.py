from sentence_transformers import SentenceTransformer, models

hub_id = 'distilbert/distilbert-base-multilingual-cased'
max_seq_length = 512

model_name = 'distilbert/distilbert-base-multilingual-cased-sent'

word_embedding_model = models.Transformer(hub_id, max_seq_length=max_seq_length)

pooling_model = models.Pooling(
                word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(), 
                pooling_mode_mean_tokens=True
                )

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model.save('./' + model_name)