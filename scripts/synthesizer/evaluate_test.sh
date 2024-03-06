# Evaluate on the test set; write to disk
python src/run.py evaluate synthesizer wiki.txt \
--reading_params_path synthesizer.finetune.params \
--eval_corpus_path birth_test_inputs.tsv \
--outputs_path synthesizer.pretrain.test.predictions