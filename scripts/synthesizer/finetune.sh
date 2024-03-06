# Finetune the model
python src/run.py finetune synthesizer wiki.txt \
--reading_params_path synthesizer.pretrain.params \
--writing_params_path synthesizer.finetune.params \
--finetune_corpus_path birth_places_train.tsv