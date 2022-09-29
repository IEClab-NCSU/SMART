rm -r FineTunedModels
mkdir FineTunedModels

for epoch in 200
do
  python run_lm_finetuning.py \
  --model_name_or_path EleutherAI/gpt-j-6B \
  --train_data_file TrainingDatasets/train_bio_10_percent.txt \
  --output_dir FineTunedModels_v2/bio_10_percent_epochs_$epoch \
  --do_train \
  --overwrite_output_dir \
  --per_gpu_train_batch_size 2 \
  --num_train_epochs $epoch \
  --save_steps 10

  python run_lm_finetuning.py \
  --model_name_or_path EleutherAI/gpt-j-6B \
  --train_data_file TrainingDatasets/train_chem_10_percent.txt \
  --output_dir FineTunedModels_v2/chem_10_percent_epochs_$epoch \
  --do_train \
  --overwrite_output_dir \
  --per_gpu_train_batch_size 2 \
  --num_train_epochs $epoch \
  --save_steps 10

  python run_lm_finetuning.py \
  --model_name_or_path EleutherAI/gpt-j-6B \
  --train_data_file TrainingDatasets/train_inspec_kdd_first_only_80_percent.txt \
  --output_dir FineTunedModels_v2/inspec_kdd_first_only_80_percent_epochs_$epoch \
  --do_train \
  --overwrite_output_dir \
  --per_gpu_train_batch_size 2 \
  --num_train_epochs $epoch \
  --save_steps 455

done
