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
from sklearn.metrics.pairwise import cosine_similarity

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
        logit = self.paraphrase_classifier(self.dropout(combined_embeddings))*5.0
        return logit.squeeze(-1)


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        

        embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        embeddings_2 = self.forward(input_ids_2, attention_mask_2)
        # combined_embeddings = torch.cat((embeddings_1, embeddings_2), dim=-1)
        # logit = self.similarity_classifier(self.dropout(combined_embeddings)) 
        # return logit.squeeze(-1)
        
        cos_sim = F.cosine_similarity(self.dropout(embeddings_1), self.dropout(embeddings_2), dim = 1)
        return cos_sim, embeddings_1,embeddings_2
    
        




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

    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

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
    # criterion_sts = nn.CosineEmbeddingLoss()


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
        task_losses = {'sts': 0}
        # total_corr = 0
        for sts_batch in tqdm(sts_train_dataloader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass and loss computation
            loss_sts = 0

            if sts_batch is not None:
                # Unpack the batches and move to device
                sts_input_ids_1, sts_attention_mask_1, sts_input_ids_2, sts_attention_mask_2, sts_labels = (sts_batch['token_ids_1'],
                    sts_batch['attention_mask_1'], sts_batch['token_ids_2'], sts_batch['attention_mask_2'], sts_batch['labels'])
                sts_input_ids_1, sts_attention_mask_1, sts_input_ids_2, sts_attention_mask_2, sts_labels = sts_input_ids_1.to(device), sts_attention_mask_1.to(device), sts_input_ids_2.to(device), sts_attention_mask_2.to(device), sts_labels.to(device)

                # sts_logits = model.predict_similarity(sts_input_ids_1, sts_attention_mask_1, sts_input_ids_2, sts_attention_mask_2)
                cosSim,e1,e2 = model.predict_similarity(sts_input_ids_1, sts_attention_mask_1, sts_input_ids_2, sts_attention_mask_2)
                # loss_sts = criterion_sts(sts_logits, sts_labels.float())
                sts_labels_normalized = (sts_labels - 2.5) / 2.5
                loss_sts = criterion_sts(cosSim, sts_labels_normalized.float())
                # loss_sts = criterion_sts(e1, e2, sts_labels_normalized.to(device))
                # cos_norm = (cosSim*2.5)+2.5
                # loss_sts = criterion_sts(cos_norm, sts_labels.float())
                task_losses['sts'] += loss_sts.item()
                # pearson_mat = np.corrcoef(cosSim.cpu().detach().numpy(), sts_labels_normalized.cpu().detach().numpy())
                # sts_corr = pearson_mat[1][0]
                # total_corr+=sts_corr
                # print("1: ",(cosSim * 2.5) + 2.5)
                # print("2: ",sts_labels)
                # print("!!: ",loss_sts)
                # print(torch.mean(torch.pow(cosSim - sts_labels_normalized, 2)))



            # Combine the losses
            loss = loss_sts

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_batches += 1

        # Evaluate the model on the development set for each task
        sts_dev_corr, sts_y_pred, sts_sent_ids = model_eval_sts(sts_dev_dataloader, model, device)

        # Print the average loss for this epoch
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss / total_batches}, Similarity: {sts_dev_corr:.3f}.")


def model_eval_sts(sts_dataloader, model, device):
    model.eval()  # switch to eval model, will turn off randomness like dropout

    with torch.no_grad():
        sts_y_true = []
        sts_y_pred = []
        sts_sent_ids = []

        # Evaluate semantic textual similarity.
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                                      batch['token_ids_2'], batch['attention_mask_2'],
                                      batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            cos_sim,e1,e2 = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            cos_sim = cos_sim.cpu().numpy()
            # y_hat = logits.flatten().cpu().numpy()
            # b_labels = b_labels.flatten().cpu().numpy()
            # cos_sim = F.cosine_similarity(e1, e2).cpu().numpy()

            y_hat = (cos_sim * 2.5) + 2.5  # Convert cosine similarity values back to the original similarity range
            b_labels = b_labels.flatten().cpu().numpy()

            # loss_sts = nn.MSELoss(torch.tensor(y_hat), torch.tensor(b_labels))
            # print(np.mean(np.power(y_hat - b_labels, 2)))
            # print("1 ", np.isnan(y_hat).any())
            # print("2 ", np.isnan(b_labels).any())
            # print("1:", e1.shape)
            # print("2:", e2.shape)
            # print("3:", e1.shape)
            # print("4:", e2.shape)
            # print("diff: ",e1-e2)
            # print("nb: ",torch.eq(e1, e2).sum().item())
            # print("5:", cosine_similarity(e1[0].cpu().numpy().reshape(-1, 1), e2[0].cpu().numpy().reshape(-1, 1)))
            # print("6: ",cos_sim[0])
            # print("1: ",y_hat)
            # print("2: ",b_labels)
            # pearson_mat = np.corrcoef(b_labels,b_labels)
            # print("!: ", pearson_mat[1][0])

            sts_y_pred.extend(y_hat)
            sts_y_true.extend(b_labels)
            sts_sent_ids.extend(b_sent_ids)

            # print("1: ",torch.eq(embed_1, embed_2).sum().item()) 
            # print("2: ",torch.eq(embed_1, embed_1).sum().item()) 

            # cos_sim2 = F.cosine_similarity(embed_1, embed_1).cpu().numpy()
            # print("1: ",np.array_equal(cos_sim, cos_sim2))
            # print("2: ",np.array_equal(cos_sim, cos_sim))

            # print("1: ",cos_sim)
            # print("2: ",cos_sim2)

            # print("1: ",embed_1)
            # print("2: ",embed_2)

        
        pearson_mat = np.corrcoef(sts_y_pred, sts_y_true)
        sts_corr = pearson_mat[1][0]

        print(f'Semantic Textual Similarity correlation: {sts_corr:.3f}')

        return sts_corr, sts_y_pred, sts_sent_ids


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

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-sts.pt'  # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)  # Modify the training function to only train for STS
    
