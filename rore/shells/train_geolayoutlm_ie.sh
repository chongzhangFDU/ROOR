export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/path/to/ROOR/rore
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

## labeling

CUDA_VISIBLE_DEVICES=0 python tasks/train_geo.py \
  --do_train true \
  --image_dir /path/to/dataset \
  --json_dir /path/to/dataset/jsons \
  --split_file_dir /path/to/dataset \
  --train_dataset_name data.train.txt \
  --valid_dataset_name data.valid.txt \
  --test_dataset_name data.test.txt \
  --label_file_name labels.txt \
  --config_json_path configs/1024.json \
  --bert_base_path /path/to/dataset/bert-base-uncased \
  --model_ckpt_path /path/to/geolayoutlm.pt \
  --save_model_dir /path/to/save_model_dir \
  --use_vision true \
  --use_segment true \
  --linking_coeff 0.5 \
  --lam 10 \
  --num_ro_layers 12 \
  --use_aux_ro true \
  --transitive_expand false \
  --batch_size 4 \
  --accumulate_grad_batches 4 \
  --max_epochs 500 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.02 \
  --monitor val_labeling_f1 \
  --log_every_n_steps 1 \
  --keep_checkpoint_max 1 \
  --patience 50 \
  --shuffle true \
  --seed 2024 \
  --gpus 1

## linking

CUDA_VISIBLE_DEVICES=0 python tasks/train_geo.py \
  --do_train true \
  --image_dir /path/to/dataset \
  --json_dir /path/to/dataset/jsons \
  --split_file_dir /path/to/dataset \
  --train_dataset_name data.train.txt \
  --valid_dataset_name data.valid.txt \
  --test_dataset_name data.test.txt \
  --label_file_name labels.txt \
  --config_json_path configs/1024.json \
  --bert_base_path /path/to/dataset/bert-base-uncased \
  --model_ckpt_path /path/to/geolayoutlm.pt \
  --save_model_dir /path/to/save_model_dir \
  --use_vision true \
  --use_segment true \
  --linking_coeff 0.5 \
  --lam 10 \
  --num_ro_layers 12 \
  --use_aux_ro true \
  --transitive_expand false \
  --batch_size 4 \
  --accumulate_grad_batches 4 \
  --max_epochs 500 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.02 \
  --monitor val_linking_f1 \
  --log_every_n_steps 1 \
  --keep_checkpoint_max 1 \
  --patience 50 \
  --shuffle true \
  --seed 2024 \
  --gpus 1

