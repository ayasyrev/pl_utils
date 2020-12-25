from pytorch_lightning.callbacks import Callback
import math
from typing import List

__all__ = ['LrSchedullerStep']


def _annealing_cos(start_value: float, end_value: float, pct: float) -> float:
    ''' Calculate value for Cosine anneal.
    Return value at pct, as pct goes from 0.0 to 1.0, from `start_value` to `end_value`. '''
    return end_value + (start_value - end_value) / 2 * (math.cos(math.pi * pct) + 1)


def annealing_cos(start_value: float, end_value: float, steps: int) -> List[float]:
    " Return list of values, Cosine anneal from `start_value` to `end_value` for `steps` steps"
    return [_annealing_cos(start_value, end_value, pct) for pct in [i / steps for i in range(steps)]]


def annealing_cos_revers(start_value: float, end_value: float, steps: int) -> List[float]:
    " Return list of values, Cosine anneal from `start_value` to `end_value` for `steps` steps"
    return [_annealing_cos(end_value, start_value, pct) for pct in [i / steps for i in range(steps)]]


def _annealing_linear(start_value: float, end_value: float, pct: float) -> float:
    ''' Calculate value for Linear anneal.
    Return value at pct, as pct goes from 0.0 to 1.0, from `start_value` to `end_value`. '''
    return start_value + (end_value - start_value) * pct


def annealing_linear(start_value: float, end_value: float, steps: int) -> List[float]:
    " Return list of values, Linear anneal from `start_value` to `end_value` for `steps` steps"
    return [_annealing_linear(start_value, end_value, pct) for pct in [i / steps for i in range(steps)]]


def annealing_linear_revers(start_value: float, end_value: float, steps: int) -> List[float]:
    " Return list of values, Linear anneal from `start_value` to `end_value` for `steps` steps"
    return [_annealing_linear(end_value, start_value, pct) for pct in [i / steps for i in range(steps)]]


annealing_fn_dict = {'cos': annealing_cos,
                     'lin': annealing_linear,
                     'cos_rev': annealing_cos_revers,
                     'lin_rev': annealing_linear_revers,
                     'step': annealing_linear}


class LrSchedullerStep(Callback):
    '''Lr scheduller. Change lr on every step by "annealing_fn" function.'''
    def __init__(self,
                 start_pct: float = 0.5,
                 annealing_pct: float = 0.25,
                 div_factor: float = 0.1,
                 annealing_fn: str = 'cos') -> None:
        self.start_pct = start_pct
        self.annealing_pct = annealing_pct
        self.div_factor = div_factor
        self.annealing_fn = annealing_fn_dict[annealing_fn]

    def on_train_start(self, trainer, pl_module):
        self.steps_at_epoch = len(trainer.train_dataloader.dataset) // trainer.train_dataloader.batch_size
        self.start_epoch = trainer.current_epoch
        total_steps = (trainer.max_epochs - trainer.current_epoch) * self.steps_at_epoch
        start_steps = int(total_steps * self.start_pct)
        annealing_steps = int(total_steps * self.annealing_pct)
        optimizer = trainer.optimizers[0]
        self.start_lr_groups = [param_group['lr'] for param_group in optimizer.param_groups]
        self.start_lr = self.start_lr_groups[0]
        self.lr_list = [self.start_lr] * start_steps
        self.lr_list.extend(self.annealing_fn(self.start_lr, self.start_lr * self.div_factor, annealing_steps))
        self.lr_list.extend([self.start_lr * self.div_factor] * (total_steps - annealing_steps))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        step_idx = (trainer.current_epoch - self.start_epoch) * self.steps_at_epoch + batch_idx
        optimizer = trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr_list[step_idx]
