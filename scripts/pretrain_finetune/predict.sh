# Evaluate on the test set; write to disk
python src/run.py evaluate vanilla wiki.txt \
--reading_params_path vanilla.finetune.params \
--eval_corpus_path birth_test_inputs.tsv \
--outputs_path vanilla.pretrain.test.predictions
