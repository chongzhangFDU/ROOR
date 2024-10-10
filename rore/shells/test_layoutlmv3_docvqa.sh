export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/path/to/ROOR/rore
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

CUDA_VISIBLE_DEVICES=0 python tasks/train_docvqa.py \
  --do_train false \
  --do_test true \
  --dataset_dir /path/to/dataset \
  --test_dataset_name train.jsonl \
  --bart_path /path/to/bart-base \
  --layoutlmv3_path /path/to/layoutlmv3-base-2048 \
  --ckpt_path /path/to/v3_bart.pt \
  --resume_from_checkpoint /path/to/some.ckpt \
  --save_model_dir /path/to/save_model_dir \
  --lam 1e-2 \
  --num_ro_layers 12 \
  --use_aux_ro true \
  --transitive_expand false \
  --box_level segment \
  --batch_size 1 \
  --log_every_n_steps 1 \
  --seed 2024 \
  --gpus 1
