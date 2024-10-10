import json
import math
import os
import copy
import datetime

import torch.multiprocessing as mp
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import init_dist_connection, rank_zero_only
from pytorch_lightning.utilities.seed import reset_seed


from argparse import ArgumentParser
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.argparse import add_argparse_args
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils.log_utils import create_logger
from utils.utils import strtobool
from utils.tensor_utils import tensor_to_strings

import itertools
import json
import os
from PIL import Image
import cv2
import numpy as np
import torch
import tqdm
from transformers import BartTokenizer, BartConfig
from model.layoutlm_v3.configuration_layoutlmv3 import LayoutLMv3Config
from model.layoutlm_v3.tokenization_layoutlmv3 import LayoutLMv3Tokenizer
from model.layoutlm_v3.generative_qa import LayoutLMv3ForConditionalGeneration
from metric.anls_metric import ANLSMetric
from dataset.docvqa_dataset import DocvqaDataset
from dataset.data_module import DocumentDataModule

class TrainingModule(pl.LightningModule):
    def __init__(self, num_samples, learning_rate=5e-5, adam_epsilon=1e-8, warmup_ratio=0., **kargs):
        super().__init__()
        self.save_hyperparameters(ignore=['collate_fn', 'tokenizer'])
        if self.global_rank == 0:
            self.local_logger = create_logger(log_dir=self.hparams.save_model_dir)
            self.local_logger.info(self.hparams)
        # 定义模型
        self.layoutlmv3_config = LayoutLMv3Config.from_pretrained(self.hparams.layoutlmv3_path)
        self.layoutlmv3_tokenizer = LayoutLMv3Tokenizer.from_pretrained(self.hparams.layoutlmv3_path)
        self.bart_config = BartConfig.from_pretrained(self.hparams.bart_path)
        self.bart_tokenizer = BartTokenizer.from_pretrained(self.hparams.bart_path)
        self.model = LayoutLMv3ForConditionalGeneration(
            config=self.layoutlmv3_config,
            bart_config=self.bart_config,
            lam=self.hparams.lam if self.hparams.use_aux_ro else None,
            num_ro_layers=self.hparams.num_ro_layers if self.hparams.use_aux_ro else None)
        # 加载模型权重
        state_dict = torch.load(self.hparams.ckpt_path)
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        ks = list(state_dict.keys())
        for k in ks:
            if k.startswith('model.'):
                new_k = k[6:]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]
        result = self.model.load_state_dict(state_dict, strict=False)
        if self.global_rank == 0:
            if result.missing_keys:
                self.local_logger.info(f"Warning: the following keys are missing in the state_dict: {result.missing_keys}")
            if result.unexpected_keys:
                self.local_logger.info(f"Warning: the following keys are unexpected in the state_dict: {result.unexpected_keys}")
        # 模型其他配置
        self.model.config.decoder_start_token_id = self.model.config.eos_token_id
        self.model.config.is_encoder_decoder = True
        self.model.config.use_cache = True
        # 其他
        self.metric = ANLSMetric() 
        

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        logits, loss = outputs.logits, outputs.loss
        steps = batch_idx
        # 这里在训练时打日志，由 log_every_n_steps 控制频率
        if self.global_rank == 0 and self.local_rank == 0 and (steps + 1) % self.trainer.log_every_n_steps == 0:
            lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
            self.local_logger.info(
                f"Epoch: {self.current_epoch}/{self.trainer.max_epochs}, "
                f"Steps: {steps}, "
                f"Learning Rate {lr_scheduler.get_last_lr()[-1]:.7f}, "
                f"Train Total Loss: {loss:.5f}, "
            )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        logits, loss = outputs.logits, outputs.loss
        # logits可能用来算metrics
        return loss

    def validation_epoch_end(self, step_outputs, split='val'):
        avg_loss = torch.tensor(0.0).to(self.device)
        n_outputs = max(1, len(step_outputs))
        for step_out in step_outputs:
            avg_loss += step_out
        self.log(f'{split}_loss', avg_loss, prog_bar=True, on_epoch=True)
        if self.global_rank == 0 and self.local_rank == 0:
            self.local_logger.info(
                f"**{split.capitalize()}** , "
                f"Epoch: {self.current_epoch}/{self.trainer.max_epochs}, "
                f"GlobalSteps: {self.global_step}, "
                f"loss: {avg_loss:.5f}, "
            )

    def decode_raw_answer(self, raw_answer):
        raw_answer = 'Answer:'.join((raw_answer.split('Answer:')[1:]))
        raw_answer = raw_answer.split('</s>')[0]
        raw_answer = raw_answer.split('<pad>')[0]
        return raw_answer

    def test_step(self, batch, batch_idx):
        batch_for_inference = {k: v for k, v in batch.items() 
            if not k.startswith('answers') and not k in ['labels']} # model.generate不支持多余的参数
        outputs = self.model.generate(**batch_for_inference, max_new_tokens=self.hparams.decoder_max_length)
        for i in range(outputs.shape[0]):
            raw_answer = self.bart_tokenizer.decode(outputs[i])
            print(raw_answer)
            print(golds)
            input()
            raw_answer = self.decode_raw_answer(raw_answer)
            golds = tensor_to_strings(
                batch['answers_raw'][i], 
                batch['answers_lengths'][i], 
                batch['answers_num_strings'][i])
            self.metric.update(raw_answer, golds)
        if batch_idx % self.hparams.log_every_n_steps == 0:
            ret = self.metric.compute()
            self.local_logger.info(
                f"**Test** , "
                f"Batch_idx: {batch_idx}, Current status: "
                f"ANLS: {ret['avg_anls']}, "
                f"Cnt: {ret['num_samples']}, "
            )
        return

    def test_epoch_end(self, step_outputs):
        ret = self.metric.compute()
        self.metric.reset()
        self.log(f'test_anls', ret['avg_anls'], prog_bar=True, on_epoch=True)
        self.log(f'cnt', ret['num_samples'], prog_bar=True, on_epoch=True)
        if self.global_rank == 0 and self.local_rank == 0:
            self.local_logger.info(
                f"**Test** , "
                f"Epoch: {self.current_epoch}/{self.trainer.max_epochs}, "
                f"GlobalSteps: {self.global_step}, "
                f"test_anls: {ret['avg_anls']:.5f}, "
                f"test_cnt: {ret['num_samples']:.5f}, "
            )
    
    def predict_step(self, batch, batch_idx):
        batch_for_inference = {k: v for k, v in batch.items() 
            if not k.startswith('answers') and not k in ['labels']} # model.generate不支持多余的参数
        outputs = self.model.generate(**batch_for_inference, max_new_tokens=self.hparams.decoder_max_length)
        answers = []
        for i in range(outputs.shape[0]):
            raw_answer = self.bart_tokenizer.decode(outputs[i])
            raw_answer = self.decode_raw_answer(raw_answer)
            answers.append(raw_answer)
        if self.global_rank == 0 and self.local_rank == 0 and batch_idx % self.hparams.log_every_n_steps == 0:
            self.local_logger.info(
                f"**Test** , "
                f"Batch_idx: {batch_idx}, "
            )
        return answers

    def configure_callbacks(self):
        callbacks = []
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        callbacks.append(EarlyStopping(monitor=self.hparams.monitor, mode="min", patience=self.hparams.patience))
        callbacks.append(
            ModelCheckpoint(monitor=self.hparams.monitor,
                            mode="min",
                            every_n_epochs=self.hparams.every_n_epochs,
                            filename='{epoch}-{step}-{val_loss:.5f}'))
        return callbacks

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        g1, g2, g3 = [], [], []
        for n, p in model.named_parameters():
            if 'lam' in n: g1.append(p)
            elif not any(nd in n for nd in no_decay): g2.append(p)
            else: g3.append(p)
        optimizer_grouped_parameters = [
            {
                "params": g1,
                "weight_decay": self.hparams.weight_decay,
                "lr": 1e-2,
            },
            {
                "params": g2,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": g3,
                "weight_decay": self.hparams.weight_decay, # 0.0
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
        else:
            raise NotImplementedError('Unknown schedule_type {self.hparams.schedule_type}')
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
    gpus = args.gpus
    if gpus > 1 and args.strategy is None:
        args.strategy = 'ddp'
    mp.set_start_method('spawn')
    pl.seed_everything(args.seed)

    layoutlmv3_config = LayoutLMv3Config.from_pretrained(args.layoutlmv3_path)
    layoutlmv3_tokenizer = LayoutLMv3Tokenizer.from_pretrained(args.layoutlmv3_path)
    bart_config = BartConfig.from_pretrained(args.bart_path)
    bart_tokenizer = BartTokenizer.from_pretrained(args.bart_path)

    index_fname_to_dataset_tmp_cache = dict()

    def index_fname_to_dataset(index_fname, is_train_val):
        if index_fname in index_fname_to_dataset_tmp_cache:
            return copy.deepcopy(index_fname_to_dataset_tmp_cache[index_fname])
        json_objs = []
        with open(os.path.join(args.dataset_dir, index_fname)) as f:
            for line in f.readlines():
                json_objs.append(json.loads(line))

        ret_dataset = DocvqaDataset(json_objs=json_objs,
                                    dataset_dir=args.dataset_dir,
                                    layoutlmv3_tokenizer=layoutlmv3_tokenizer,
                                    bart_tokenizer=bart_tokenizer,
                                    layoutlmv3_config=layoutlmv3_config,
                                    bart_config=bart_config,
                                    encoder_max_length=args.encoder_max_length,
                                    decoder_max_length=args.decoder_max_length,
                                    box_level=args.box_level,
                                    use_aux_ro=args.use_aux_ro,
                                    transitive_expand=args.transitive_expand,
                                    is_train_val=is_train_val
                                    )
        index_fname_to_dataset_tmp_cache[index_fname] = ret_dataset
        return ret_dataset

    train_dataset, valid_dataset, test_dataset, num_samples = None, None, None, None
    if args.train_dataset_name != 'None' and args.do_train:
        train_dataset = index_fname_to_dataset(args.train_dataset_name, is_train_val=True)
        num_samples = len(train_dataset)
    if args.valid_dataset_name != 'None' and args.do_train:
        valid_dataset = index_fname_to_dataset(args.valid_dataset_name, is_train_val=True)
    if args.test_dataset_name != 'None' and (args.do_test or args.do_predict):
        test_dataset = index_fname_to_dataset(args.test_dataset_name, is_train_val=False)
        if num_samples == None:
            num_samples = len(test_dataset)

    data_module = DocumentDataModule(train_dataset=train_dataset,
                                     valid_dataset=valid_dataset,
                                     test_dataset=test_dataset,
                                     batch_size=args.batch_size,
                                     val_test_batch_size=1, # 不支持batch decode
                                     shuffle=args.shuffle,
                                     )

    # trainer

    model = TrainingModule(**vars(args), num_samples=num_samples)
    logger = TensorBoardLogger(save_dir=args.save_model_dir, name='')


    # class CustomPlugin(pl.plugins.training_type.DDPPlugin):

    #     def setup_distributed(self):
    #         reset_seed()

    #         # determine which process we are and world size
    #         self.set_world_ranks()

    #         # set warning rank
    #         rank_zero_only.rank = self.global_rank

    #         # set up server using proc 0's ip address
    #         # try to init for 20 times at max in case ports are taken
    #         # where to store ip_table
    #         init_dist_connection(self.cluster_environment, self.torch_distributed_backend, timeout=datetime.timedelta(seconds=7200))


    trainer = Trainer.from_argparse_args(args,
                                         weights_save_path=args.save_model_dir,
                                         logger=logger,
                                         enable_progress_bar=False,
                                         resume_from_checkpoint=args.resume_from_checkpoint,
                                         plugins=[LightningEnvironment()],) # strategy=CustomPlugin(),

    if args.do_train:
        trainer.fit(model, data_module)

    if args.do_test:
        trainer.test(model, data_module)

    if args.do_predict:
        raise NotImplementedError

if __name__ == '__main__':
    # 添加conflict_handler，防止和trainer参数冲突
    parser = ArgumentParser(conflict_handler='resolve')
    parser = Trainer.add_argparse_args(parser)
    parser = DocumentDataModule.add_argparse_args(parser)

    # Data Hyperparameters
    parser.add_argument('--dataset_dir', default='/path/to/FUNSD', type=str)
    parser.add_argument('--train_dataset_name', default='None', type=str)
    parser.add_argument('--valid_dataset_name', default='None', type=str)
    parser.add_argument('--test_dataset_name', default='None', type=str)
    parser.add_argument('--shuffle', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='训练集是否shuffle',
                        default=True)

    # Model Hyperparameters

    parser.add_argument('--bart_path', default='', type=str)
    parser.add_argument('--layoutlmv3_path', default='', type=str)
    parser.add_argument('--ckpt_path', default='ckpts/v3_bart.pt', type=str)
    parser.add_argument('--resume_from_checkpoint', default=None, type=str)
    parser.add_argument('--encoder_max_length', default=1536, type=int)
    parser.add_argument('--decoder_max_length', default=512, type=int)

    # Basic Training Control

    parser.add_argument('--do_train', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='do train',
                        default=True)
    parser.add_argument('--do_test', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='do test',
                        default=False)
    parser.add_argument('--do_predict', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='do test',
                        default=False)
    parser.add_argument('--precision', default=32, type=int, )
    parser.add_argument('--num_nodes', default=1, type=int, )
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--strategy', default=None, type=str)
    parser.add_argument('--lam', default=0.1, type=float)
    parser.add_argument('--num_ro_layers', default=24, type=int)
    parser.add_argument('--use_aux_ro', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='是否使用阅读顺序信号', default=False)
    parser.add_argument('--transitive_expand', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='阅读顺序信号是否增加传递性', default=False)
    parser.add_argument('--box_level', default='segment', type=str)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    parser.add_argument('--val_test_batch_size', default=None, type=int)
    parser.add_argument('--schedule_type', default='cosine', type=str, help='constant, linear, cosine', )
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--warmup_ratio', default=0, type=float)
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--gradient_clip_val', default=1.0, type=float)
    parser.add_argument('--gradient_clip_algorithm', default='norm', type=str)
    parser.add_argument('--sync_batchnorm', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='Synchronize batch norm layers between process groups/whole world.', default=True)
    parser.add_argument('--monitor', default='val_loss', type=str)  # val_linking_f1
    parser.add_argument('--save_model_dir', default='lightning_logs', type=str)
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