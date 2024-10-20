export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/path/to/ROOR/rore
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

CUDA_VISIBLE_DEVICES=0 python tasks/train_v3_ner.py \
  --do_train true \
  --image_dir /path/to/dataset \
  --json_dir /path/to/dataset/jsons \
  --split_file_dir /path/to/dataset \
  --train_dataset_name data.train.txt \
  --valid_dataset_name data.val.txt \
  --label_file_name labels.txt \
  --pretrained_model_path /path/to/dataset/layoutlmv3-base-2048 \
  --save_model_dir /path/to/dataset/save_model_dir \
  --box_level segment \
  --lam 1e-2 \
  --num_ro_layers 12 \
  --use_aux_ro true \
  --transitive_expand false \
  --batch_size 2 \
  --accumulate_grad_batches 8 \
  --max_epochs 500 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.02 \
  --log_every_n_steps 1 \
  --keep_checkpoint_max 1 \
  --patience 50 \
  --shuffle true \
  --seed 2024 \
  --gpus 1