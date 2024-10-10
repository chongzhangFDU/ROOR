export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/mnt/user/zhang_chong/reading-order-relation-prediction


CUDA_VISIBLE_DEVICES=0 python src/tasks/pl_training.py \
  --do_train false \
  --do_predict true \
  --image_dir /path/to/FUNSD/ \
  --json_dir /path/to/FUNSD/jsons \
  --split_file_dir /path/to/FUNSD \
  --test_dataset_name data.all.txt \
  --bbox_level segment \
  --unit_type segment \
  --max_num_units 256 \
  --pretrained_model_path /path/to/layoutlmv3-large-2048 \
  --ckpt_path /path/to/some.ckpt \
  --save_model_dir /path/to/save_model_dir \
  --predict_output_dir /path/to/FUNSD/jsons_pred \
  --gpus 1