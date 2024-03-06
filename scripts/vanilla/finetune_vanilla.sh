CUDA_VISIBLE_DEVICES=1 python src/run.py finetune vanilla wiki.txt \
--writing_params_path vanilla.model.params \
--finetune_corpus_path birth_places_train.tsv

# finetune without pretraining 