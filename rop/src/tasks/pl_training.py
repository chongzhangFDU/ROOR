import json
import math
import os
import copy

import torch.multiprocessing as mp
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.argparse import add_argparse_args
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from src.data.ro_dataset import RoDataset
from src.data.ro_datamodule import RODataModule
from src.model.layoutlm_v3.configuration_layoutlmv3 import LayoutLMv3Config
from src.model.layoutlm_v3.modeling_layoutlmv3 import LayoutLMv3ForRORelation
from src.model.layoutlm_v3.tokenization_layoutlmv3_fast import LayoutLMv3TokenizerFast
from src.metric.ro_metric import PRFA
from src.utils.log_utils import create_logger
from src.utils.utils import strtobool
from src.utils.prediction_utils import decode

torch.backends.cudnn.enabled = False


class ROTrainingModule(pl.LightningModule):
    def __init__(self,
                 num_samples: int,
                 learning_rate: float = 2e-5,
                 adam_epsilon: float = 1e-6,
                 warmup_ratio: float = 0.05,
                 **kargs):
        super().__init__()
        self.save_hyperparameters(ignore=['collate_fn', 'tokenizer'])

        # 模型定义
        self.config = LayoutLMv3Config.from_pretrained(self.hparams.pretrained_model_path, output_hidden_states=True)
        self.model = LayoutLMv3ForRORelation.from_pretrained(config=self.config,
                                                             pretrained_model_name_or_path=self.hparams.pretrained_model_path,
                                                             ignore_mismatched_sizes=False)
        # 其他定义
        self.metric = PRFA()
        if self.global_rank == 0:
            self.local_logger = create_logger(log_dir=self.hparams.save_model_dir)
            self.local_logger.info(self.hparams)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        steps = batch_idx
        # 这里在训练时打日志，由 log_every_n_steps 控制频率
        if self.global_rank == 0 and self.local_rank == 0 and (steps + 1) % self.trainer.log_every_n_steps == 0:
            lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
            self.local_logger.info(
                f"Epoch: {self.current_epoch}/{self.trainer.max_epochs}, "
                f"Steps: {steps}, "
                f"Learning Rate {lr_scheduler.get_last_lr()[-1]:.7f}, "
                f"Train Loss: {loss:.5f}"
            )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        logits, loss = outputs.logits, outputs.loss
        logits, grid_labels, num_units = logits.detach().cpu(), batch['grid_labels'].detach().cpu(), batch['num_units'].detach().cpu()
        self.metric.update(logits, grid_labels, num_units)
        return loss

    def validation_epoch_end(self, step_outputs, split='val'):
        metric_results = self.metric.compute()
        loss = torch.stack(step_outputs).mean()
        self.log(f'{split}_loss', loss, prog_bar=True, on_epoch=True)
        self.log(f'{split}_f1', metric_results['f1'], prog_bar=True, on_epoch=True)
        self.log(f'{split}_p', metric_results['precision'], prog_bar=True, on_epoch=True)
        self.log(f'{split}_r', metric_results['recall'], prog_bar=True, on_epoch=True)
        self.log(f'{split}_acc', metric_results['accuracy'], prog_bar=True, on_epoch=True)
        if self.global_rank == 0 and self.local_rank == 0:
            self.local_logger.info(
                f"**{split.capitalize()}** , "
                f"Epoch: {self.current_epoch}/{self.trainer.max_epochs}, "
                f"GlobalSteps: {self.global_step}, "
                f"loss: {loss:.5f}, "
                f"f1: {metric_results['f1']:.5f}, "
                f"precision: {metric_results['precision']:.5f}, "
                f"recall: {metric_results['recall']:.5f}, "
                f"acc: {metric_results['accuracy']:.5f}, "
                f"num_samples: {metric_results['num_samples']}"
            )
        self.metric.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, step_outputs):
        return self.validation_epoch_end(step_outputs, split='test')

    def predict_step(self, batch, batch_idx):
        outputs = self(**batch)
        logits, loss = outputs.logits, outputs.loss
        logits, num_units = logits.detach().cpu(), batch['num_units'].detach().cpu()
        preds = []
        for logit, n in zip(logits, num_units):
            n = n.item()
            logit = logit.squeeze(0)[:n, :n]
            edges = dict()
            # 初步挑出所有positive predictions，然后交给去环逻辑（decode）
            for i in range(n):
                for j in range(n):
                    if logit[i][j] > 0: edges[(i, j)] = logit[i][j].item()
            preds.append(decode(n, edges))
        return preds # 输出unit idx pairs，但是无法对应回unit（seg/word）的原始id

    def configure_callbacks(self):
        callbacks = []
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        callbacks.append(EarlyStopping(monitor="val_f1", mode="max", patience=self.hparams.patience))
        callbacks.append(
            ModelCheckpoint(monitor="val_f1",
                            mode="max",
                            every_n_epochs=self.hparams.every_n_epochs,
                            filename='{epoch}-{step}-{val_f1:.5f}-{val_acc:.5f}'))
        return callbacks

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                      lr=self.hparams.learning_rate,
                                      eps=self.hparams.adam_epsilon)
        num_warmup_steps = int(self.total_steps * self.hparams.warmup_ratio)
        if self.hparams.schedule_type == 'constant':
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
            )
        elif self.hparams.schedule_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif self.hparams.schedule_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps,
            )
        else: raise NotImplementedError('Unknown schedule_type {self.hparams.schedule_type}')
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def setup(self, stage=None):
        if stage != "fit":
            return
        num_samples = self.hparams.num_samples
        batch_size = self.hparams.batch_size

        steps = math.ceil(num_samples / batch_size / max(1, self.trainer.devices))
        # Calculate total steps
        ab_steps = int(steps / self.trainer.accumulate_grad_batches)
        self.total_steps = int(ab_steps * self.trainer.max_epochs)
        if self.global_rank == 0 and self.local_rank == 0:
            self.local_logger.info(f"- num_samples is: {num_samples}")
            self.local_logger.info(f"- max_epochs is: {self.trainer.max_epochs}")
            self.local_logger.info(f"- total_steps is: {self.total_steps}")
            self.local_logger.info(f"- batch size (1 gpu) is: {batch_size}")
            self.local_logger.info(f"- devices(gpu) num is: {max(1, self.trainer.devices)}")
            self.local_logger.info(f"- accumulate_grad_batches is: {self.trainer.accumulate_grad_batches}")

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        return add_argparse_args(cls, parent_parser, **kwargs)


def main(args):
    
    # settings

    gpus = args.gpus
    if gpus > 1 and args.strategy is None:
        args.strategy = 'ddp'
    mp.set_start_method('spawn')
    pl.seed_everything(args.seed)
    
    # model

    config = LayoutLMv3Config.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_path)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_path)
    model = LayoutLMv3ForRORelation.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_path,
        head_size=args.head_size,
        dropout=args.dropout)
    
    # datamodule

    index_fname_to_dataset_tmp_cache = dict()

    def index_fname_to_dataset(index_fname):
        if index_fname in index_fname_to_dataset_tmp_cache:
            return copy.deepcopy(index_fname_to_dataset_tmp_cache[index_fname])
        json_objs = []
        with open(os.path.join(args.split_file_dir, index_fname)) as f:
            for fname in f.readlines():
                with open(os.path.join(args.json_dir, fname.strip())) as f1:
                    json_objs.append(json.load(f1))
        ret_dataset =  RoDataset(dataset=json_objs,
                         img_dir=args.image_dir,
                         tokenizer=tokenizer,
                         max_x=args.max_x,
                         max_y=args.max_y,
                         xy_same_scale=False,
                         box_level=args.bbox_level,
                         unit_type=args.unit_type,
                         max_num_units=args.max_num_units,
                         max_seq_len=tokenizer.model_max_length)
        index_fname_to_dataset_tmp_cache[index_fname] = ret_dataset
        return ret_dataset

    train_dataset, valid_dataset, test_dataset = None, None, None
    num_samples = None
    if args.train_dataset_name != 'None' and args.do_train:
        train_dataset = index_fname_to_dataset(args.train_dataset_name)
        num_samples = len(train_dataset)
    if args.valid_dataset_name != 'None' and args.do_train:
        valid_dataset = index_fname_to_dataset(args.valid_dataset_name)
    if args.test_dataset_name != 'None' and (args.do_test or args.do_predict):
        test_dataset = index_fname_to_dataset(args.test_dataset_name)

    data_module = RODataModule(train_dataset=train_dataset,
                              valid_dataset=valid_dataset,
                              test_dataset=test_dataset,
                              batch_size=args.batch_size,
                              val_test_batch_size=args.val_test_batch_size,
                              shuffle=args.shuffle,
                              )
    
    # trainer

    model = ROTrainingModule(**vars(args), num_samples=num_samples)
    logger = TensorBoardLogger(save_dir=args.save_model_dir, name='')
    trainer = Trainer.from_argparse_args(args,
                                         weights_save_path=args.save_model_dir,
                                         logger=logger,
                                         enable_progress_bar=False,
                                         plugins=[LightningEnvironment()])
   
    # train

    if args.do_train:
        trainer.fit(model, data_module)

    if args.do_test:
        trainer.test(model, data_module, ckpt_path='best' if args.ckpt_path is None else args.ckpt_path)

    if args.do_predict:
        predictions = trainer.predict(
            model, data_module, ckpt_path='best' if args.ckpt_path is None else args.ckpt_path)
        predictions = [pred for prediction_batch in predictions for pred in prediction_batch]
        if not os.path.exists(args.predict_output_dir):
            os.mkdir(args.predict_output_dir)
        with open(os.path.join(args.split_file_dir, args.test_dataset_name)) as f:
            for fname, pred in zip(f.readlines(), predictions):
                fname = fname.strip()
                with open(os.path.join(args.json_dir, fname)) as f1:
                    json_obj = json.load(f1)
                    unit_idx_to_res_id = dict()
                    if args.unit_type == 'segment':
                        start_idx = 0
                        for segment in json_obj['document']:
                            unit_idx_to_res_id[start_idx] = segment['id']
                            start_idx += 1
                    elif args.unit_type == 'word':
                        start_idx = 0
                        for segment in json_obj['document']:
                            for word in segment['words']:
                                unit_idx_to_res_id[start_idx] = word['id']
                                start_idx += 1
                    json_obj['ro_linkings'] = sorted([[unit_idx_to_res_id[i], unit_idx_to_res_id[j]] for i, j in pred])
                    if 'label_linkings' in json_obj:
                        json_obj['label_linkings'].sort()
                with open(os.path.join(args.predict_output_dir, fname), 'w', encoding='utf-8') as fw:
                    json.dump(json_obj, fw, ensure_ascii=False)
                    

if __name__ == '__main__':
    # 添加conflict_handler，防止和trainer参数冲突
    parser = ArgumentParser(conflict_handler='resolve')
    parser = Trainer.add_argparse_args(parser)
    parser = RODataModule.add_argparse_args(parser)

    # Data Hyperparameters
    parser.add_argument('--image_dir', default='./datasets/ROOR', type=str)
    parser.add_argument('--json_dir', default='./datasets/ROOR/jsons', type=str)
    parser.add_argument('--split_file_dir', default='./datasets/ROOR', type=str)
    parser.add_argument('--train_dataset_name', default='None', type=str)
    parser.add_argument('--valid_dataset_name', default='None', type=str)
    parser.add_argument('--test_dataset_name', default='None', type=str)
    parser.add_argument('--bbox_level', default='segment', type=str, help='word or segment')
    parser.add_argument('--unit_type', default='segment', type=str, help='word or segment')
    parser.add_argument('--max_num_units', default=256, type=int, )
    parser.add_argument('--max_x', default=1023, type=int)
    parser.add_argument('--max_y', default=1023, type=int)
    parser.add_argument('--shuffle', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='训练集是否shuffle',
                        default=True)

    # Model Hyperparameters
    parser.add_argument('--pretrained_model_path', default='./models/layoutlmv3-base-2048',
                        type=str)
    parser.add_argument('--head_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)

    # Basic Training Control

    parser.add_argument('--do_train', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='do train',
                        default=True)
    parser.add_argument('--do_test', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='do test',
                        default=False)
    parser.add_argument('--do_predict', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='do test',
                        default=False)
    parser.add_argument('--predict_output_dir', default='./datasets/FUNSD/jsons_pred', type=str)
   

    parser.add_argument('--precision', default=32, type=int, )
    parser.add_argument('--num_nodes', default=1, type=int, )
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--strategy', default=None, type=str)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--accumulate_grad_batches', default=2, type=int)
    parser.add_argument('--val_test_batch_size', default=None, type=int)
    parser.add_argument('--schedule_type', default='cosine', type=str, help='constant, linear, cosine',)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--patience', default=50, type=int)

    parser.add_argument('--save_model_dir', default='lightning_logs', type=str)
    parser.add_argument('--ckpt_path', default=None, type=str)
    parser.add_argument('--log_every_n_steps', default=1, type=int)
    parser.add_argument('--val_check_interval', default=1.0, type=float)  # int时多少个steps跑验证集,float 按照比例算
    parser.add_argument('--every_n_epochs', default=1, type=int)
    parser.add_argument('--keep_checkpoint_max', default=1, type=int)
    parser.add_argument('--deploy_path', default='', type=str)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--detect_anomaly', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='是否开启detect',
                        default=False)
    args = parser.parse_args()
    print(args)
    
    main(args)