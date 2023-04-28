import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import zip_longest
from types import SimpleNamespace
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask,model_eval_multitask


TQDM_DISABLE=True

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_classifier = nn.Linear(config.hidden_size, 5)

        #这两行代码中，self.bert.config.hidden_size * 2是因为我们在处理两个句子对的嵌入时，需要将它们拼接在一起。
        self.paraphrase_classifier = nn.Linear(config.hidden_size * 2, 1)
        self.similarity_classifier = nn.Linear(config.hidden_size * 2, 1)
       

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO

        # Pass input tokens and attention masks through BERT model
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Get the [CLS] token hidden states
        cls_output = bert_outputs['pooler_output']
 
        return cls_output


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        # ### TODO
        # raise NotImplementedError

        embeddings = self.forward(input_ids, attention_mask)
        logits = self.sentiment_classifier(self.dropout(embeddings))
        return logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        embeddings_2 = self.forward(input_ids_2, attention_mask_2)
        combined_embeddings = torch.cat((embeddings_1, embeddings_2), dim=-1)
        logit = self.paraphrase_classifier(self.dropout(combined_embeddings))
        return logit.squeeze(-1)


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        embeddings_2 = self.forward(input_ids_2, attention_mask_2)
        combined_embeddings = torch.cat((embeddings_1, embeddings_2), dim=-1)
        logit = self.similarity_classifier(self.dropout(combined_embeddings))       
        return logit.squeeze(-1)
        




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)
    # Define loss functions
    criterion_sst = nn.CrossEntropyLoss()
    criterion_para = nn.BCEWithLogitsLoss()
    criterion_sts = nn.MSELoss()

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # Initialize the best development scores for each task
    best_dev_score_sst = 0
    best_dev_score_para = 0
    best_dev_score_sts = 0
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        task_losses = {'sst': 0, 'para': 0, 'sts': 0}
        gradient_accumulation_counter = 0

        # for sst_batch, para_batch, sts_batch in zip(sst_train_dataloader, para_train_dataloader, sts_train_dataloader):
        for sst_batch, para_batch, sts_batch in zip_longest(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, fillvalue=None):         
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass and loss computation
            loss_sst, loss_para, loss_sts = 0, 0, 0
            if sst_batch is not None:
                # Unpack the batches and move to device
                sst_input_ids, sst_attention_mask, sst_labels = (sst_batch['token_ids'], sst_batch['attention_mask'], sst_batch['labels'])
                sst_input_ids, sst_attention_mask, sst_labels = sst_input_ids.to(device), sst_attention_mask.to(device), sst_labels.to(device)

                sst_logits = model.predict_sentiment(sst_input_ids, sst_attention_mask)
                loss_sst = criterion_sst(sst_logits, sst_labels)
                task_losses['sst'] += loss_sst.item()

            if para_batch is not None:
                # Unpack the batches and move to device
                para_input_ids_1, para_attention_mask_1, para_input_ids_2, para_attention_mask_2, para_labels = (para_batch['token_ids_1'], 
                    para_batch['attention_mask_1'], para_batch['token_ids_2'], para_batch['attention_mask_2'], para_batch['labels'])
                para_input_ids_1, para_attention_mask_1, para_input_ids_2, para_attention_mask_2, para_labels = para_input_ids_1.to(device), para_attention_mask_1.to(device), para_input_ids_2.to(device), para_attention_mask_2.to(device), para_labels.to(device)

                para_logits = model.predict_paraphrase(para_input_ids_1, para_attention_mask_1, para_input_ids_2, para_attention_mask_2)
                loss_para = criterion_para(para_logits, para_labels.float())
                task_losses['para'] += loss_para.item()

            if sts_batch is not None:
                # Unpack the batches and move to device
                sts_input_ids_1, sts_attention_mask_1, sts_input_ids_2, sts_attention_mask_2, sts_labels = (sts_batch['token_ids_1'], 
                    sts_batch['attention_mask_1'], sts_batch['token_ids_2'], sts_batch['attention_mask_2'], sts_batch['labels'])
                sts_input_ids_1, sts_attention_mask_1, sts_input_ids_2, sts_attention_mask_2, sts_labels = sts_input_ids_1.to(device), sts_attention_mask_1.to(device), sts_input_ids_2.to(device), sts_attention_mask_2.to(device), sts_labels.to(device)

                sts_logits = model.predict_similarity(sts_input_ids_1, sts_attention_mask_1, sts_input_ids_2, sts_attention_mask_2)
                loss_sts = criterion_sts(sts_logits, sts_labels.float())
                task_losses['sts'] += loss_sts.item()

            # Combine the losses
            loss = loss_sst + loss_para + loss_sts

            # Backward pass
            # loss.backward()

            # Backward pass
            loss_sst.backward(retain_graph=True)
            grad_sst = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_sst.append(param.grad.clone().detach())

            loss_para.backward(retain_graph=True)
            grad_para = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_para.append(param.grad.clone().detach())

            loss_sts.backward()
            grad_sts = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_sts.append(param.grad.clone().detach())

            # Gradient surgery
            # Perform gradient projection here, assuming grad_sst, grad_para, and grad_sts have the gradients for each task
            # Modify the following code to match the number of tasks you have and the desired projection order
            for i, (g_sst, g_para, g_sts) in enumerate(zip_longest(grad_sst, grad_para, grad_sts, fillvalue=None)):
                if g_sst is None:
                    g_sst = torch.zeros_like(g_para)
                if g_para is None:
                    g_para = torch.zeros_like(g_para)
                if g_sts is None:
                    g_sts = torch.zeros_like(g_para)
                
                param.grad = torch.zeros_like(g_para)

                g_sst_flat = g_sst.view(-1)
                g_para_flat = g_para.view(-1)
                g_sts_flat = g_sts.view(-1)

                g_sst_projected = g_sst_flat - g_para_flat * (g_sst_flat.dot(g_para_flat)) / g_para_flat.norm() ** 2
                g_sst_projected = g_sst_projected - g_sts_flat * (g_sst_projected.dot(g_sts_flat)) / g_sts_flat.norm() ** 2

                g_para_projected = g_para_flat - g_sst_flat * (g_para_flat.dot(g_sst_flat)) / g_sst_flat.norm() ** 2
                g_para_projected = g_para_projected - g_sts_flat * (g_para_projected.dot(g_sts_flat)) / g_sts_flat.norm() ** 2

                g_sts_projected = g_sts_flat - g_sst_flat * (g_sts_flat.dot(g_sst_flat)) / g_sst_flat.norm() ** 2
                g_sts_projected = g_sts_projected - g_para_flat * (g_sts_projected.dot(g_para_flat)) / g_para_flat.norm() ** 2

                g_combined = g_sst_projected + g_para_projected + g_sts_projected
                param.grad = g_combined.view(param.grad.shape)


            # # Gradient clipping   optional
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)

            # Gradient accumulation
            gradient_accumulation_counter += 1
            if gradient_accumulation_counter % args.gradient_accumulation_steps == 0:
                # Perform optimization step
                optimizer.step()
                # Zero the gradients
                optimizer.zero_grad()
                gradient_accumulation_counter = 0

            total_loss += loss.item()
            total_batches+=1
         # Print the average loss for the current epoch
        print(f'Epoch {epoch + 1} Loss: {total_loss / total_batches:.4f}')
        print(f'SST Loss: {task_losses["sst"] / total_batches:.4f}')
        print(f'Para Loss: {task_losses["para"] / total_batches:.4f}')
        print(f'STS Loss: {task_losses["sts"] / total_batches:.4f}')

        # Evaluate the model on the development set for each task
        # para_dev_acc, *_ , sst_dev_acc, *_ , sts_dev_corr, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
        para_dev_acc, para_y_pred, para_sent_ids, sst_dev_acc, sst_y_pred, sst_sent_ids, sts_dev_corr, sts_y_pred, sts_sent_ids = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        # Print the average loss for this epoch
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss / total_batches}")

        # Save the model if the development scores are better than the best so far
        improved = False
        if sst_dev_acc > best_dev_score_sst:
            best_dev_score_sst = sst_dev_acc
            improved = True
        if para_dev_acc > best_dev_score_para:
            best_dev_score_para = para_dev_acc
            improved = True
        if sts_dev_corr > best_dev_score_sts:
            best_dev_score_sts = sts_dev_corr
            improved = True

        if improved:
            save_model(model, optimizer, args, config, args.filepath)
            print(f"New best development scores: Sentiment: {best_dev_score_sst:.3f}, Paraphrase: {best_dev_score_para:.3f}, Similarity: {best_dev_score_sts:.3f}. Model saved.")




def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    
    #  Add the new argument for gradient clipping
    parser.add_argument("--clip_value", type=float, default=1.0,
                        help="Gradient clipping threshold value")
    # for gradient acc.
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients before updating weights')
    #python train.py --gradient_accumulation_steps 4 

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
