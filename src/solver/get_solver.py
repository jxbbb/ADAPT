from .optimization import AdamW, WarmupLinearSchedule
from .optimization import WarmupConstantSchedule, WarmupCosineSchedule


def get_optimizer(model, weight_decay, learning_rate, adam_epsilon):
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    return AdamW(grouped_parameters, lr=learning_rate, eps=adam_epsilon)


def get_scheduler(optimizer, scheduler_type, warmup_steps, t_total):
    if scheduler_type == "constant":
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmup_steps)
    elif scheduler_type == "linear":
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=warmup_steps, t_total=t_total
        )
    elif scheduler_type == "cosine":
        scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps=warmup_steps, t_total=t_total
        )
    else:
        raise ValueError("Unknown scheduler type: {}".format(scheduler_type))
    return scheduler

