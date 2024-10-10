export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/path/to/ROOR/rore
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

CUDA_VISIBLE_DEVICES=0 python tasks/train_geo.py \
  --do_train false \
  --do_test true \
  --image_dir /path/to/dataset \
  --json_dir /path/to/dataset/jsons \
  --split_file_dir /path/to/dataset \
  --train_dataset_name geo.train.txt \
  --valid_dataset_name geo.valid.txt \
  --test_dataset_name geo.test.txt \
  --label_file_name labels.txt \
  --config_json_path configs/1024.json \
  --bert_base_path /path/to/bert-base-uncased \
  --model_ckpt_path /path/to/geolayoutlm_ckpt.pt \
  --save_model_dir /path/to/save_model_dir \
  --use_vision true \
  --linking_coeff 0.0 \
  --num_ro_layers 12 \
  --use_aux_ro true \
  --transitive_expand false \
  --batch_size 4 \
  --accumulate_grad_batches 4 \
  --learning_rate 1e-5 \
  --log_every_n_steps 1 \
  --keep_checkpoint_max 1 \
  --shuffle true \
  --seed 2024 \
  --gpus 1