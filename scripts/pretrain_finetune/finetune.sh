# Finetune the model
python src/run.py finetune vanilla wiki.txt \
--reading_params_path vanilla.pretrain.params \
--writing_params_path vanilla.finetune.params \
--finetune_corpus_path birth_places_train.tsv