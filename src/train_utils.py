import numpy as np
import evaluate
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
import torch
from contrastive.evaluation import get_auroc, get_fpr_95
from torch.utils.tensorboard import SummaryWriter
import os

writer = SummaryWriter(f'logs/BERT_contastive/')
roc_auc = evaluate.load("roc_auc")
f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")


def merge_keys(l, keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = []
        for i in l:
            new_dict[key] += i[key]
    return new_dict


def evaluate_ood(model, dataloader, ood, config):
    keys = ['softmax', 'maha', 'cosine', 'energy']

    in_scores = []
    for batch in dataloader:
        model.eval()
        batch = {key: value.to(config.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch)
            in_scores.append(ood_keys)
    in_scores = merge_keys(in_scores, keys)

    out_scores = []
    for batch in ood:
        model.eval()
        batch = {key: value.to(config.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch)
            out_scores.append(ood_keys)
    out_scores = merge_keys(out_scores, keys)

    outputs = {}
    for key in keys:
        ins = np.array(in_scores[key], dtype=np.float64)
        outs = np.array(out_scores[key], dtype=np.float64)
        inl = np.ones_like(ins).astype(np.int64)
        outl = np.zeros_like(outs).astype(np.int64)
        scores = np.concatenate([ins, outs], axis=0)
        labels = np.concatenate([inl, outl], axis=0)

        auroc, fpr_95 = get_auroc(labels, scores), get_fpr_95(labels, scores)

        outputs[key + "_auroc"] = auroc
        outputs[key + "_fpr95"] = fpr_95
    return outputs


def train(model, train_dataloader, dev_dataloader, test_dataloader, ood, config):
    path = f'runs/{config.project_name}/'
    if not os.path.exists(path):
        os.mkdir(path)
        
    total_steps = int(len(train_dataloader) * config.num_epoches)
    warmup_steps = int(total_steps * config.warmup_ratio)

    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=config.adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    def detect_ood():
        model.prepare_ood(train_dataloader)
        results = evaluate_ood(model, dev_dataloader, ood, config)

        writer.add_scalar('test/softmax_auroc', results['softmax_auroc'], num_steps)
        writer.add_scalar('test/softmax_fpr95', results['softmax_fpr95'], num_steps)
        writer.add_scalar('test/maha_auroc', results['maha_auroc'], num_steps)
        writer.add_scalar('test/maha_fpr95', results['maha_fpr95'], num_steps)
        writer.add_scalar('test/cosine_auroc', results['cosine_auroc'], num_steps)
        writer.add_scalar('test/cosine_fpr95', results['cosine_fpr95'], num_steps)
        writer.add_scalar('test/energy_auroc', results['energy_auroc'], num_steps)
        writer.add_scalar('test/energy_fpr95', results['energy_fpr95'], num_steps)

        print(results)
            # wandb.log(results, step=num_steps)

    num_steps = 0
    for epoch in range(int(config.num_epoches)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            batch = {key: value.to(config.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss, cos_loss = outputs[0], outputs[1]
            loss.backward()
            num_steps += 1
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            writer.add_scalar('train/Loss', loss.item(), num_steps)
            writer.add_scalar('train/Loss_coss', cos_loss.item(), num_steps)
            

        results = validation(model, dev_dataloader, config, tag="dev")
        writer.add_scalar('eval/f1_macro', results['f1_macro'], num_steps)
        writer.add_scalar('eval/f1_micro', results['f1_micro'], num_steps)
        writer.add_scalar('eval/f1_weighted', results['f1_weighted'], num_steps)
        writer.add_scalar('eval/accuracy', results['accuracy'], num_steps)

        print(results)
        
        detect_ood()
        path = f'runs/{config.project_name}/checkpoint-{num_steps}/'
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(model, path + 'model.pt')

    
def validation(model, dataloader, config, tag="train"):
    def compute_metrics(preds, labels):
        predictions = np.argmax(preds, axis=1)
        f1_m = f1.compute(predictions=predictions, references=labels, average='macro')["f1"]
        f1_micro = f1.compute(predictions=predictions, references=labels, average='micro')["f1"]
        f1_w = f1.compute(predictions=predictions, references=labels, average='weighted')["f1"]
        accuracy_m = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        
        return {"f1_macro": f1_m,
                "f1_micro": f1_micro, 
                "f1_weighted": f1_w,
                "accuracy": accuracy_m}
        
    label_list, logit_list = [], []
    for step, batch in enumerate(tqdm(dataloader)):
        model.eval()
        labels = batch["labels"].detach().cpu().numpy()
        batch = {key: value.to(config.device) for key, value in batch.items()}
        batch["labels"] = None
        outputs = model(**batch)
        logits = outputs[0].detach().cpu().numpy()
        label_list.append(labels)
        logit_list.append(logits)
    preds = np.concatenate(logit_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    
    results = compute_metrics(preds, labels)
    
    return results