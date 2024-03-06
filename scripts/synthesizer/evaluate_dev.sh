# Evaluate on the dev set; write to disk
python src/run.py evaluate synthesizer wiki.txt \
--reading_params_path synthesizer.finetune.params \
--eval_corpus_path birth_dev.tsv \
--outputs_path synthesizer.pretrain.dev.predictions