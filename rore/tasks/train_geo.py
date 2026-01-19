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
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils.log_utils import create_logger
from utils.utils import strtobool

from transformers import BertTokenizer
from dataset.dataset import DocumentDataset
from dataset.data_module import DocumentDataModule
from model.geolayoutlm.modeling_geolayoutlm import GeoLayoutLMVIEModel
from model.geolayoutlm.configuration_geolayoutlm import GeoLayoutLMConfig
from metric.geo_metric import GeoMetric

class TrainingModule(pl.LightningModule):
    def __init__(self, eval_kwargs, num_samples, n_classes, learning_rate=5e-5, adam_epsilon=1e-8, warmup_ratio=0., **kargs):
        super().__init__()
        self.save_hyperparameters(ignore=['collate_fn', 'tokenizer'])
        with open(self.hparams.config_json_path) as f:
            config_json = json.load(f)
        self.config = GeoLayoutLMConfig(**config_json)
        self.config.backbone_config['lam'] = self.hparams.lam
        self.config.backbone_config['num_ro_layers'] = self.hparams.num_ro_layers
        self.config.n_classes = n_classes
        self.model = GeoLayoutLMVIEModel(
            config=self.config,
            bert_base_path=self.hparams.bert_base_path,
            model_ckpt_path=self.hparams.model_ckpt_path,
            use_vision=self.hparams.use_vision,
            linking_coeff=self.hparams.linking_coeff)
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.bert_base_path, do_lower_case=True)
        self.metric = GeoMetric(tokenizer=self.tokenizer, eval_kwargs=eval_kwargs) # dump_dir='./results'
        if self.global_rank == 0:
            self.local_logger = create_logger(log_dir=self.hparams.save_model_dir)
            self.local_logger.info(self.hparams)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        head_outputs, loss_dict = self(batch)
        loss, labeling_loss, linking_loss = \
            loss_dict["total_loss"], loss_dict["labeling_loss"], loss_dict["linking_loss"]
        steps = batch_idx
        # 这里在训练时打日志，由 log_every_n_steps 控制频率
        if self.global_rank == 0 and self.local_rank == 0 and (steps + 1) % self.trainer.log_every_n_steps == 0:
            lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
            self.local_logger.info(
                f"Epoch: {self.current_epoch}/{self.trainer.max_epochs}, "
                f"Steps: {steps}, "
                f"Learning Rate {lr_scheduler.get_last_lr()[-1]:.7f}, "
                f"Train Total Loss: {loss:.5f}, "
                f"Labeling Loss: {labeling_loss:.5f}, "
                f"Linking Loss: {linking_loss:.5f}"
            )
        return loss

    def validation_step(self, batch, batch_idx):
        head_outputs, loss_dict = self(batch)
        self.metric.update(batch, head_outputs, loss_dict)
        return loss_dict

    def validation_epoch_end(self, step_outputs, split='val'):
        avg_loss = torch.tensor(0.0).to(self.device)
        avg_labeling_loss = torch.tensor(0.0).to(self.device)
        avg_linking_loss = torch.tensor(0.0).to(self.device)
        n_outputs = max(1, len(step_outputs))
        for step_out in step_outputs:
            avg_loss += step_out["total_loss"]
            avg_labeling_loss += step_out["labeling_loss"]
            avg_linking_loss += step_out["linking_loss"]
        metric_results = self.metric.compute()
        self.log(f'{split}_loss', avg_loss, prog_bar=True, on_epoch=True)
        self.log(f'{split}_labeling_loss', avg_labeling_loss, prog_bar=True, on_epoch=True)
        self.log(f'{split}_linking_loss', avg_linking_loss, prog_bar=True, on_epoch=True)
        self.log(f'{split}_labeling_f1', metric_results['labeling']['f1'], prog_bar=True, on_epoch=True)
        self.log(f'{split}_labeling_p', metric_results['labeling']['precision'], prog_bar=True, on_epoch=True)
        self.log(f'{split}_labeling_r', metric_results['labeling']['recall'], prog_bar=True, on_epoch=True)
        self.log(f'{split}_linking_f1', metric_results['linking']['f1'], prog_bar=True, on_epoch=True)
        self.log(f'{split}_linking_p', metric_results['linking']['precision'], prog_bar=True, on_epoch=True)
        self.log(f'{split}_linking_r', metric_results['linking']['recall'], prog_bar=True, on_epoch=True)
        if self.global_rank == 0 and self.local_rank == 0:
            self.local_logger.info(
                f"**{split.capitalize()}** , "
                f"Epoch: {self.current_epoch}/{self.trainer.max_epochs}, "
                f"GlobalSteps: {self.global_step}, "
                f"loss: {avg_loss:.5f}, "
                f"labeling_loss: {avg_labeling_loss:.5f}, "
                f"linking_loss: {avg_linking_loss:.5f}, "
                f"labeling_f1: {metric_results['labeling']['f1']:.5f}, "
                f"labeling_p: {metric_results['labeling']['precision']:.5f}, "
                f"labeling_r: {metric_results['labeling']['recall']:.5f}, "
                f"linking_f1: {metric_results['linking']['f1']:.5f}, "
                f"linking_p: {metric_results['linking']['precision']:.5f}, "
                f"linking_r: {metric_results['linking']['recall']:.5f}"
            )
        self.metric.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, step_outputs):
        return self.validation_epoch_end(step_outputs, split='test')

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_callbacks(self):
        if self.hparams.monitor is None:
            self.hparams.monitor = 'val_labeling_f1'
        callbacks = []
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        callbacks.append(EarlyStopping(monitor=self.hparams.monitor, mode="max", patience=self.hparams.patience))
        callbacks.append(
            ModelCheckpoint(monitor=self.hparams.monitor,
                            mode="max",
                            every_n_epochs=self.hparams.every_n_epochs,
                            filename='{epoch}-{step}-{val_labeling_f1:.5f}-{val_linking_f1:.5f}'))
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
    with open(args.config_json_path) as f:
        config_json = json.load(f)
    config = GeoLayoutLMConfig(**config_json)
    if args.label_file_name is not None:
        with open(os.path.join(args.split_file_dir, args.label_file_name)) as f:
            class_names = [l.strip() for l in f.readlines()] + ['O']
        config.n_classes = 2 * len(class_names) - 1
    else:
        class_names = None
    

    tokenizer = BertTokenizer.from_pretrained(args.bert_base_path, do_lower_case=True)

    # data module

    index_fname_to_dataset_tmp_cache = dict()

    def index_fname_to_dataset(index_fname):
        if index_fname in index_fname_to_dataset_tmp_cache:
            return copy.deepcopy(index_fname_to_dataset_tmp_cache[index_fname])
        json_objs = []
        with open(os.path.join(args.split_file_dir, index_fname)) as f:
            for fname in f.readlines():
                with open(os.path.join(args.json_dir, fname.strip())) as f1:
                    json_objs.append(json.load(f1))
        
        ret_dataset = DocumentDataset(dataset=json_objs,
                                      img_dir=args.image_dir,
                                      tokenizer=tokenizer,
                                      max_block_num=args.max_block_num, 
                                      max_seq_length=config.backbone_config['max_position_embeddings'], 
                                      img_h=config.backbone_config['image_size'][0], 
                                      img_w=config.backbone_config['image_size'][1],
                                      use_segment=args.use_segment,
                                      real_scan=args.real_scan,
                                      use_aux_ro=args.use_aux_ro,
                                      transitive_expand=args.transitive_expand,
                                      class_names=class_names
                                      )
        index_fname_to_dataset_tmp_cache[index_fname] = ret_dataset
        return ret_dataset

    train_dataset, valid_dataset, test_dataset = None, None, None
    if args.train_dataset_name != 'None' and args.do_train:
        train_dataset = index_fname_to_dataset(args.train_dataset_name)
        eval_kwargs = {'bio_class_names': train_dataset.bio_class_names}
        num_samples = len(train_dataset)
    if args.valid_dataset_name != 'None' and args.do_train:
        valid_dataset = index_fname_to_dataset(args.valid_dataset_name)
    if args.test_dataset_name != 'None' and (args.do_test or args.do_predict):
        test_dataset = index_fname_to_dataset(args.test_dataset_name)

    data_module = DocumentDataModule(train_dataset=train_dataset,
                               valid_dataset=valid_dataset,
                               test_dataset=test_dataset,
                               batch_size=args.batch_size,
                               val_test_batch_size=args.val_test_batch_size,
                               shuffle=args.shuffle,
                               )

    # trainer

    model = TrainingModule(**vars(args), eval_kwargs=eval_kwargs, num_samples=num_samples, n_classes=config.n_classes)
    logger = TensorBoardLogger(save_dir=args.save_model_dir, name='')
    trainer = Trainer.from_argparse_args(args,
                                         weights_save_path=args.save_model_dir,
                                         logger=logger,
                                         enable_progress_bar=False,
                                         plugins=[LightningEnvironment()])


    if args.do_train:
        trainer.fit(model, data_module)

    if args.do_test:
        trainer.test(model, data_module, save_ckpt_path='best' if args.save_ckpt_path is None else args.save_ckpt_path)

    if args.do_predict:
        raise NotImplementedError


if __name__ == '__main__':
    # 添加conflict_handler，防止和trainer参数冲突
    parser = ArgumentParser(conflict_handler='resolve')
    parser = Trainer.add_argparse_args(parser)
    parser = DocumentDataModule.add_argparse_args(parser)

    # Data Hyperparameters
    parser.add_argument('--image_dir', default='/path/to/dataset', type=str)
    parser.add_argument('--json_dir', default='/path/to/dataset/jsons', type=str)
    parser.add_argument('--split_file_dir', default='/path/to/dataset', type=str)
    parser.add_argument('--train_dataset_name', default='None', type=str)
    parser.add_argument('--valid_dataset_name', default='None', type=str)
    parser.add_argument('--test_dataset_name', default='None', type=str)
    parser.add_argument('--label_file_name', default='labels.txt', type=str)
    parser.add_argument('--shuffle', type=lambda x: bool(strtobool(x)), nargs='?', const=True, help='训练集是否shuffle',
                        default=True)

    # Model Hyperparameters

    parser.add_argument('--config_json_path', default='configs/config.json', type=str)
    parser.add_argument('--bert_base_path', default='configs/config.json', type=str)
    parser.add_argument('--model_ckpt_path', default='ckpts/geolayoutlm.pt', type=str)

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
    parser.add_argument('--max_block_num', default=160, type=int, help='entity linking时最大实体个数')
    parser.add_argument('--use_vision', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='entity linking是否使用视觉信号', default=True)
    parser.add_argument('--linking_coeff', default=0.5, type=float, help='linking loss的权重，0-1之间，0:全是labeling，1:全是linking')
    parser.add_argument('--use_segment', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='是否使用segment框', default=True)
    parser.add_argument('--real_scan', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='是否扰动segment框', default=False)
    parser.add_argument('--lam', default=0.1, type=float)
    parser.add_argument('--num_ro_layers', default=12, type=int)
    parser.add_argument('--use_aux_ro', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='是否使用阅读顺序信号', default=False)
    parser.add_argument('--transitive_expand', type=lambda x: bool(strtobool(x)), nargs='?', const=True,
                        help='阅读顺序信号是否增加传递性', default=False)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--accumulate_grad_batches', default=2, type=int)
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
    parser.add_argument('--monitor', default='val_labeling_f1', type=str) # val_linking_f1
    parser.add_argument('--save_model_dir', default='lightning_logs', type=str)
    parser.add_argument('--save_ckpt_path', default=None, type=str)
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