from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class DatasetSwitchHook(Hook):
    def __init__(self, last, train_file, val_file):
        self.last = last
        self._restart_dataloader = False
        self._has_switched = False
        self.train_file = train_file
        self.val_file = val_file
        
    def before_train_epoch(self, runner):
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        if runner.max_epochs - epoch <= self.last and not self.has_switched:
            runner.logger.info('Switch train dataset now!')
            
            train_loader.dataset.ann_file = self.val_file
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
        else:
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True
    
    def before_val_epoch(self, runner):
        epoch = runner.epoch
        val_loader = runner.val_dataloader
        if runner.max_epochs - epoch <= self.last and not self.has_switched:
            runner.logger.info('Switch val dataset now!')
            
            val_loader.dataset.ann_file = self.train_file
            if hasattr(val_loader, 'persistent_workers'
                       ) and val_loader.persistent_workers is True:
                val_loader._DataLoader__initialized = False
                val_loader._iterator = None
                self._restart_dataloader = True
            self._has_switched = True
        else:
            if self._restart_dataloader:
                val_loader._DataLoader__initialized = True