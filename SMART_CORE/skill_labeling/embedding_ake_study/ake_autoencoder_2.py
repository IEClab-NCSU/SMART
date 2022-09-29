import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from keyphrase_extractor import *
import math


def prep_data(data, _min, _max):

    # normalize the data
    # data_df = pd.DataFrame(data)
    # print(data_df.describe().T)

    # data_df = (data_df - _min) / (_max - _min)
    # print(data_df.describe().T)

    # data = data_df.values

    # convert the data to torch tensors
    data = torch.from_numpy(data)
    data = data.float()

    return data


def make_ake_dataset(doc_embeddings, cand_embeddings_uni, cand_embeddings_other, verbose=False):
    doc_embeddings = np.array(doc_embeddings)
    doc_embeddings = doc_embeddings.reshape(doc_embeddings.shape[0], doc_embeddings.shape[2])

    flat_cand_uni = [embedding for cand in cand_embeddings_uni for embedding in cand]
    flat_cand_other = [embedding for cand in cand_embeddings_other for embedding in cand]
    cand_embeddings = flat_cand_uni + flat_cand_other
    cand_embeddings = np.array(cand_embeddings)

    data_df = pd.DataFrame(doc_embeddings)
    _min, _max = data_df.iloc[:, :].min(), data_df.iloc[:, :].max()

    cand_embed_unigram = np.array(cand_embeddings_uni)
    cand_embed_other = np.array(cand_embeddings_other)

    norm_doc_embeddings = prep_data(doc_embeddings, _min, _max)

    norm_cand_embeddings = prep_data(cand_embeddings, _min, _max)

    norm_cand_embed_unigram = [prep_data(np.array(cand_embed), _min, _max) for cand_embed in cand_embed_unigram]
    norm_cand_embed_other = [prep_data(np.array(cand_embed), _min, _max) for cand_embed in cand_embed_other]

    norm_combined_embeddings = prep_data(np.concatenate((doc_embeddings, cand_embeddings), axis=0), _min, _max)

    if verbose:
        print('Shape and Type of Data: {} : {}'.format(doc_embeddings.shape, doc_embeddings.dtype))

    return norm_combined_embeddings, norm_doc_embeddings, norm_cand_embeddings, norm_cand_embed_unigram, norm_cand_embed_other


class AutoEncoder(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        torch.manual_seed(2022)
        self.encoder_hidden_layer = nn.Linear(input_dim, 320)
        self.encoder_hidden_layer_2 = nn.Linear(320, 128)
        self.encoder_hidden_layer_3 = nn.Linear(128, 64)
        self.encoder_hidden_layer_4 = nn.Linear(64, 32)
        self.decoder_hidden_layer = nn.Linear(32, 64)
        self.decoder_hidden_layer_2 = nn.Linear(64, 128)
        self.decoder_hidden_layer_3 = nn.Linear(128, 320)
        self.decoder_hidden_layer_4 = nn.Linear(320, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        op = self.encoder_hidden_layer(x)
        op = self.relu(op)
        # op = self.dropout(op)
        op = self.encoder_hidden_layer_2(op)
        op = self.relu(op)
        # op = self.dropout(op)
        op = self.encoder_hidden_layer_3(op)
        # op = self.relu(op)
        # op = self.dropout(op)
        op = self.encoder_hidden_layer_4(op)
        latent = self.relu(op)

        op = self.decoder_hidden_layer(latent)
        # op = self.relu(op)
        # op = self.dropout(op)
        op = self.decoder_hidden_layer_2(op)
        op = self.relu(op)
        # op = self.dropout(op)
        op = self.decoder_hidden_layer_3(op)
        op = self.relu(op)
        # op = self.dropout(op)
        y = self.decoder_hidden_layer_4(op)
        # y = self.sigmoid(op)

        return latent, y


def train_network(model, optimizer, loss_function, num_epochs, batch_size, train_embeds, doc_embeds, cand_embeds):
    model.train()
    train_loss_across_batches = []
    doc_loss_across_batches = []
    cand_loss_across_batches = []
    train_loss_across_epochs = []
    doc_loss_across_epochs = []
    cand_loss_across_epochs = []
    for epoch in range(num_epochs):
        for i in range(0, train_embeds.shape[0], batch_size):

            # Extract train batch from X and Y
            input_data = train_embeds[i:min(train_embeds.shape[0], i+batch_size)]
            labels = doc_embeds[i:min(doc_embeds.shape[0], i+batch_size)]

            optimizer.zero_grad()

            _, output_data = model(input_data)

            batch_loss = loss_function(output_data, labels)

            batch_loss.backward()

            optimizer.step()

            _, y_pred = model(train_embeds)
            train_loss = loss_function(y_pred, doc_embeds)
            train_loss = math.sqrt(train_loss.item())
            train_loss_across_batches.append(train_loss)

            _, doc_pred = model(doc_embeds)
            doc_loss = loss_function(doc_pred, doc_embeds)
            doc_loss = math.sqrt(doc_loss.item())
            doc_loss_across_batches.append(doc_loss)

            _, cand_pred = model(cand_embeds)
            cand_loss = loss_function(cand_pred, cand_embeds)
            cand_loss = math.sqrt(cand_loss.item())
            cand_loss_across_batches.append(cand_loss)

        _, y_pred = model(train_embeds)
        train_epoch_loss = loss_function(y_pred, doc_embeds)
        train_epoch_loss = math.sqrt(train_epoch_loss.item())
        train_loss_across_epochs.append(train_epoch_loss)
        
        _, y_pred = model(doc_embeds)
        doc_epoch_loss = loss_function(y_pred, doc_embeds)
        doc_epoch_loss = math.sqrt(doc_epoch_loss.item())
        doc_loss_across_epochs.append(doc_epoch_loss)
        
        _, y_pred = model(cand_embeds)
        cand_epoch_loss = loss_function(y_pred, cand_embeds)
        cand_epoch_loss = math.sqrt(cand_epoch_loss.item())
        cand_loss_across_epochs.append(cand_epoch_loss)

        print('Epoch: {}, Train Loss: {:.4f}, Document Loss: {:.4f}, Candidate Loss: {:.4f}'.format(epoch + 1, train_epoch_loss, doc_epoch_loss, cand_epoch_loss))

    return train_loss_across_batches, train_loss_across_epochs, doc_loss_across_batches, doc_loss_across_epochs, cand_loss_across_batches, cand_loss_across_epochs


'''def calc_cand_loss(cand_embeddings):
    flat_cand_uni = [embedding for cand in cand_uni for embedding in cand]
    flat_cand_other = [embedding for cand in cand_other for embedding in cand]
    cand_embeddings = flat_cand_uni + flat_cand_other'''


def create_and_run_autoencoder_model(train_embeds, doc_embeds, cand_embeds, cand_embed_uni, cand_embed_other, filepath, plot_lc=True):
    if torch.cuda.is_available():
    # print('GPU In Use')
        train_embeds = train_embeds.cuda()
        doc_embeds = train_embeds.cuda()
        cand_embeds = cand_embeds.cuda()
    # else:
    # print('No GPU')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder(train_embeds.shape[1]).to(device)

    # model = AutoEncoder(train_embeds.shape[1])

    print('Training Data shape: {}'.format(train_embeds.shape))

    learning_rate = 0.05
    num_epochs = 1500
    batch_size = 16
    loss_function = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)

    train_batches, train_epochs, doc_batches, doc_epochs, cand_batches, cand_epochs = train_network(model, optimizer, loss_function, num_epochs, batch_size, train_embeds, doc_embeds, cand_embeds)

    if plot_lc == 'batch':
        # plt.plot(train_batches, label='Train Loss')
        plt.plot(doc_batches, label='Document Loss')
        plt.plot(cand_batches, label='Candidate Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(filepath)
    else:
        # plt.plot(train_epochs, label='Train Loss')
        plt.plot(doc_epochs, label='Document Loss')
        plt.plot(cand_epochs, label='Candidate Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(filepath)

    latent_reps_doc, y_pred = model(doc_embeds)
    latent_reps_doc = latent_reps_doc.cpu().detach().numpy()

    latent_cand_uni = []
    total_loss = 0.0
    num_candidates = 0
    for candidates_uni in cand_embed_uni:
        latent_cands_uni = []
        for candidate_uni in candidates_uni:
            if torch.cuda.is_available():
                    candidate_uni = candidate_uni.cuda()
            num_candidates += 1
            latent_rep_uni, y_pred_uni = model(candidate_uni)
            latent_cands_uni.append(latent_rep_uni.cpu().detach().numpy().tolist())
            cand_loss_uni = loss_function(y_pred_uni, candidate_uni)
            cand_loss_uni = math.sqrt(cand_loss_uni.item())
            total_loss += cand_loss_uni
        latent_cand_uni.append(latent_cands_uni)

    latent_cand_other = []
    for candidates_other in cand_embed_other:
        latent_cands_other = []
        for candidate_other in candidates_other:
            if torch.cuda.is_available():
                    candidate_other = candidate_other.cuda()
            num_candidates += 1
            latent_rep_other, y_pred_other = model(candidate_other)
            latent_cands_other.append(latent_rep_other.cpu().detach().numpy().tolist())
            cand_loss_other = loss_function(y_pred_other, candidate_other)
            cand_loss_other = math.sqrt(cand_loss_other.item())
            total_loss += cand_loss_other
        latent_cand_other.append(latent_cands_other)

    candidate_loss = total_loss / num_candidates
    print('Overall Candidate Loss: {}:{}'.format(total_loss, candidate_loss))

    return latent_reps_doc, latent_cand_uni, latent_cand_other


'''if __name__ == '__main__':
    x, cand_embed_uni, cand_embed_other = make_ake_dataset('inspec', 'parsing', 'sbert-mean-pooling')
    latent_doc, latent_cand_uni, latent_cand_other = create_and_run_autoencoder_model(x, cand_embed_uni, cand_embed_other)
    print('Example Latent Representation (1st Document): {}'.format(latent_doc[0]))
    print('Example Latent Representation (1st Unigram Candidate for 1st Document): {}'.format(latent_cand_uni[0][0]))
    print('Example Latent Representation (1st Other Candidate for 1st Document): {}'.format(latent_cand_other[0][0]))'''
