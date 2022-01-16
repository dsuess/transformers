#!/bin/sh

PYTHONPATH=src USE_TF=0 CUDA_VISIBLE_DEVICES=0 python ./examples/pytorch/translation/run_translation.py \
  --model_name_or_path t5-small --output_dir /tmp/zero3 --overwrite_output_dir --max_train_samples 10 --max_eval_samples 1600 --max_source_length 512 --max_target_length 512  --do_train --num_train_epochs 1 --per_device_train_batch_size 4 --\per_device_eval_batch_size 16 --learning_rate 3e-3 --warmup_steps 8 --predict_with_generate --logging_steps 0 --save_steps 2 --eval_steps 1 --group_by_length --adafactor --dataset_name wmt16 --dataset_config ro-en --source_lang en --target_lang ro --source_prefix "translate English to Romanian: " --do_eval --logging_steps 1 $@
