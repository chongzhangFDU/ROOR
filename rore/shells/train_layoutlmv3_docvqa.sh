export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/path/to/ROOR/rore
export CUDA_DEVICE_ORDER="PCI_BUS_ID"


CUDA_VISIBLE_DEVICES=0 python tasks/train_docvqa.py \
  --do_train true \
  --dataset_dir /path/to/dataset \
  --train_dataset_name train.jsonl \
  --valid_dataset_name train.jsonl \
  --bart_path /path/to/dataset/bart-base \
  --layoutlmv3_path /path/to/dataset/layoutlmv3-base-2048 \
  --ckpt_path /path/to/some.ckpt \
  --save_model_dir /path/to/save_model_dir \
  --schedule_type cosine \
  --lam 0.1 \
  --num_ro_layers 24 \
  --use_aux_ro true \
  --transitive_expand false \
  --box_level segment \
  --batch_size 1 \
  --accumulate_grad_batches 1 \
  --max_epochs 200 \
  --every_n_epochs 20 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.02 \
  --log_every_n_steps 1 \
  --keep_checkpoint_max 1 \
  --patience 50 \
  --shuffle true \
  --seed 2024 \
  --gpus 1
