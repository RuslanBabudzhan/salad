import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler

import utils
from models import helper
from utils.grad_cache import grad_cache_backward


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
        #---- Backbone
        backbone_arch='resnet50',
        backbone_config={},
        
        #---- Aggregator
        agg_arch='ConvAP',
        agg_config={},
        
        #---- Train hyperparameters
        lr=0.03, 
        optimizer='sgd',
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },
        
        #----- Loss
        loss_name='MultiSimilarityLoss', 
        miner_name='MultiSimilarityMiner', 
        miner_margin=0.1,
        faiss_gpu=False,
        grad_cache_chunk_size=0,
    ):
        super().__init__()
        self.automatic_optimization = False
        if grad_cache_chunk_size < 0:
            raise ValueError('grad_cache_chunk_size must be nonnegative')
        self.grad_cache_chunk_size = grad_cache_chunk_size

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config
        
        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        if agg_arch.lower() == 'salad':
            agg_config = dict(agg_config)
            agg_config.setdefault('num_channels', self.backbone.num_channels)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

        # For validation in Lightning v2.0.0
        self.val_outputs = []
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr, 
                weight_decay=self.weight_decay, 
                momentum=self.momentum
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        

        if self.lr_sched.lower() == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_args['milestones'], gamma=self.lr_sched_args['gamma'])
        elif self.lr_sched.lower() == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args['T_max'])
        elif self.lr_sched.lower() == 'linear':
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_args['start_factor'],
                end_factor=self.lr_sched_args['end_factor'],
                total_iters=self.lr_sched_args['total_iters']
            )

        else:
            raise ValueError(f'Unknown scheduler: {self.lr_sched}')

        return [optimizer], [scheduler]
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        if not torch.isfinite(descriptors).all():
            raise ValueError('Nonfinite descriptors')
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        self.log('b_acc', batch_acc, on_step=True, on_epoch=True, batch_size=len(labels))
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        # MegaLoc supplies separate subsets; legacy GSV-Cities supplies one pair.
        subsets = batch if isinstance(batch, dict) else {'train': batch}
        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=self.device)
        for name, (images, labels) in subsets.items():
            images = images.reshape(-1, *images.shape[-3:])
            labels = labels.reshape(-1)
            if self.grad_cache_chunk_size:
                loss = grad_cache_backward(
                    self, images, labels, self.loss_function,
                    self.grad_cache_chunk_size, self.manual_backward,
                )
            else:
                descriptors = self(images.to(self.device, non_blocking=True)).float()
                with torch.autocast(device_type=self.device.type, enabled=False):
                    loss = self.loss_function(descriptors, labels.to(self.device))
                    self.manual_backward(loss)
                loss = loss.detach()
            total += loss
            self.log(f'loss/{name}', loss, on_step=True, on_epoch=False, batch_size=len(labels))

        scaler = getattr(self.trainer.precision_plugin, 'scaler', None)
        scale = scaler.get_scale() if scaler is not None else None
        optimizer.step()
        # A reduced AMP scale means the optimizer skipped an overflowing update.
        if scaler is None or scaler.get_scale() >= scale:
            self.lr_schedulers().step()
        self.log('loss', total, prog_bar=True, on_step=True, on_epoch=False, batch_size=1)
        return {'loss': total}

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.trainer.training and isinstance(batch, dict):
            # Keep all image groups on CPU; transfer only the active chunk.
            return batch
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def on_train_start(self):
        if self.trainer.world_size != 1:
            raise ValueError('This training loop currently supports one device')

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()
    
    def on_validation_epoch_start(self):
        # reset the outputs list
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]
    
    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.val_outputs

        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            pitts_dict = utils.get_validation_recalls(
                r_list=r_list, 
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')

        # reset the outputs list
        self.val_outputs = []
