from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import pytorch_lightning as pl
from torch import nn
import torch
from torch.utils.data import DataLoader
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from os import listdir
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import kuzu
import multiprocessing
import time
import numpy as np

from layer import RGTLayer


NUM_EPOCHS = 1
LOADER_BATCH_SIZE = 256

    
class RGTDetector(nn.Module):
    def __init__(self, args):
        super(RGTDetector, self).__init__()
    
        self.lr = args.lr
        self.l2_reg = args.l2_reg
    
        self.in_linear_numeric = nn.Linear(args.numeric_num, int(args.linear_channels/4), bias=True)
        self.in_linear_bool = nn.Linear(args.cat_num, int(args.linear_channels/4), bias=True)
        self.in_linear_tweet = nn.Linear(args.tweet_channel, int(args.linear_channels/4), bias=True)
        self.in_linear_des = nn.Linear(args.des_channel, int(args.linear_channels/4), bias=True)
        self.linear1 = nn.Linear(args.linear_channels, args.linear_channels)

        self.RGT_layer1 = RGTLayer(num_edge_type=2, in_channel=args.linear_channels, out_channel=args.out_channel, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)
        self.RGT_layer2 = RGTLayer(num_edge_type=2, in_channel=args.linear_channels, out_channel=args.out_channel, trans_heads=args.trans_head, semantic_head=args.semantic_head, dropout=args.dropout)

        self.out1 = torch.nn.Linear(args.out_channel, 64)
        self.out2 = torch.nn.Linear(64, 2)

        self.drop = nn.Dropout(args.dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.ReLU = nn.LeakyReLU()
        
        self.init_weight()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2_reg, amsgrad=False)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=16, eta_min=0)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, batch):
        # Extract features from the batch
        prop_features = batch['user'].prop_features
        cat_features = batch['user'].cat_features
        tweet_features = batch['user'].tweet_features
        des_features = batch['user'].des_features

        # Process features through the layers
        user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
        user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
        user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
        user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
        
        user_features = torch.cat((user_features_numeric, user_features_bool, user_features_tweet, user_features_des), dim=1)
        user_features = self.drop(self.ReLU(self.linear1(user_features)))

        following_edge_index = batch['user', 'following', 'user'].edge_index
        follows_edge_index = batch['user', 'follows', 'user'].edge_index
        edge_index_list = [following_edge_index, follows_edge_index]

        user_features = self.ReLU(self.RGT_layer1(user_features, edge_index_list))
        user_features = self.ReLU(self.RGT_layer2(user_features, edge_index_list))

        user_features = self.drop(self.ReLU(self.out1(user_features)))
        output = self.out2(user_features)

        return output

    def training_step(self, train_batch):
        batch_size = train_batch['user'].batch_size

        prop_features = train_batch['user'].prop_features
        cat_features = train_batch['user'].cat_features
        tweet_features = train_batch['user'].tweet_features
        des_features = train_batch['user'].des_features
        labels = train_batch['user'].y[0:batch_size]

        user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
        user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
        user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
        user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
        
        user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
        user_features = self.drop(self.ReLU(self.linear1(user_features)))

        following_edge_index = train_batch['user', 'following', 'user'].edge_index
        follows_edge_index = train_batch['user', 'follows', 'user'].edge_index
        edge_index_list = [following_edge_index, follows_edge_index]

        user_features = self.ReLU(self.RGT_layer1(user_features, edge_index_list))
        user_features = self.ReLU(self.RGT_layer2(user_features, edge_index_list))

        user_features = self.drop(self.ReLU(self.out1(user_features)))
        pred = self.out2(user_features[0:batch_size])
        loss = self.CELoss(pred, labels)

        return loss
    
    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            cat_features = test_batch['user'].cat_features
            prop_features = test_batch['user'].prop_features
            tweet_features = test_batch['user'].tweet_features
            des_features = test_batch['user'].des_features
            
            label = test_batch['user'].y

            following_edge_index = test_batch['user', 'following', 'user'].edge_index
            follows_edge_index = test_batch['user', 'follows', 'user'].edge_index

            # Concatenate the edge indices
            edge_index = torch.cat((following_edge_index, follows_edge_index), dim=1)

            # Create edge type tensor
            # Assuming 'following' edges are type 0 and 'follows' edges are type 1
            following_edge_type = torch.zeros(following_edge_index.size(1), dtype=torch.long)
            follows_edge_type = torch.ones(follows_edge_index.size(1), dtype=torch.long)

            # Concatenate the edge types
            edge_type = torch.cat((following_edge_type, follows_edge_type), dim=0)
            
            user_features_numeric = self.drop(self.ReLU(self.in_linear_numeric(prop_features)))
            user_features_bool = self.drop(self.ReLU(self.in_linear_bool(cat_features)))
            user_features_tweet = self.drop(self.ReLU(self.in_linear_tweet(tweet_features)))
            user_features_des = self.drop(self.ReLU(self.in_linear_des(des_features)))
            
            user_features = torch.cat((user_features_numeric,user_features_bool,user_features_tweet,user_features_des), dim = 1)
            user_features = self.drop(self.ReLU(self.linear1(user_features)))

            user_features = self.ReLU(self.RGT_layer1(user_features, edge_index, edge_type))
            user_features = self.ReLU(self.RGT_layer2(user_features, edge_index, edge_type))

            user_features = self.drop(self.ReLU(self.out1(user_features)))
            pred = self.out2(user_features)
            
            pred_binary = torch.argmax(pred, dim=1)

            acc = accuracy_score(label.cpu(), pred_binary.cpu())
            f1 = f1_score(label.cpu(), pred_binary.cpu())
            precision =precision_score(label.cpu(), pred_binary.cpu())
            recall = recall_score(label.cpu(), pred_binary.cpu())
            auc = roc_auc_score(label.cpu(), pred[:,1].cpu())

            self.log("acc", acc)
            self.log("f1",f1)
            self.log("precision", precision)
            self.log("recall", recall)
            self.log("auc", auc)

            print("acc: {} \t f1: {} \t precision: {} \t recall: {} \t auc: {}".format(acc, f1, precision, recall, auc))


parser = argparse.ArgumentParser(description="Reproduction of Heterogeneity-aware Bot detection with Relational Graph Transformers")
parser.add_argument("--path", type=str, default="./", help="dataset path")
parser.add_argument("--numeric_num", type=int, default=5, help="dataset path")
parser.add_argument("--linear_channels", type=int, default=128, help="linear channels")
parser.add_argument("--cat_num", type=int, default=3, help="catgorical features")
parser.add_argument("--des_channel", type=int, default=768, help="description channel")
parser.add_argument("--tweet_channel", type=int, default=768, help="tweet channel")
parser.add_argument("--out_channel", type=int, default=128, help="description channel")
parser.add_argument("--dropout", type=float, default=0.5, help="description channel")
parser.add_argument("--trans_head", type=int, default=8, help="description channel")
parser.add_argument("--semantic_head", type=int, default=8, help="description channel")
parser.add_argument("--batch_size", type=int, default=1, help="description channel") # was init 256
parser.add_argument("--epochs", type=int, default=50, help="description channel")
parser.add_argument("--lr", type=float, default=1e-3, help="description channel")
parser.add_argument("--l2_reg", type=float, default=3e-5, help="description channel")
parser.add_argument("--random_seed", type=int, default=None, help="random")


def evaluate_model(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            # Forward pass
            outputs = model(batch)  # Adjust this line according to how your model produces outputs
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['user'].y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


if __name__ == "__main__":
    global args
    args = parser.parse_args()

    train_idx = torch.arange(0, 8278)
    valid_idx = torch.arange(8278, 10643)
    test_idx = torch.arange(10643, 11826)
    db = kuzu.Database("TwiBot-20")
    conn = kuzu.Connection(db, num_threads=multiprocessing.cpu_count())
    feature_store, graph_store = db.get_torch_geometric_remote_backend(multiprocessing.cpu_count())
    train_loader = NeighborLoader(data=(feature_store, graph_store), num_neighbors={('user', 'follows', 'user'): [6, 6, 6], ('user', 'following', 'user'): [6,6,6]}, input_nodes=('user', train_idx), batch_size=128, shuffle=True, filter_per_worker=False)
    valid_loader = NeighborLoader(data=(feature_store, graph_store), num_neighbors={('user', 'follows', 'user'): [6, 6, 6], ('user', 'following', 'user'): [6,6,6]}, input_nodes=('user', valid_idx), batch_size=1, shuffle=False, filter_per_worker=False)

    model = RGTDetector(args)
    model.train()

    print("Training Model with DEFAULT KUZUDB BACKEND")

    timings = []

    NUM_EPOCHS = 5
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        accum_loss = 0.0
        n_elements = 0
        data_loading_start = time.time()
        total_data_loading_time = 0
        for batch in train_loader:
            data_loading_end = time.time()
            load_time = (data_loading_end - data_loading_start)
            print("Batch load time: ", load_time)
            total_data_loading_time += load_time
            model.optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            model.optimizer.step()
            val_accuracy = evaluate_model(model, valid_loader)
            print(f"Validation Accuracy at end of epoch {epoch}: {val_accuracy}")
            accum_loss += loss
            n_elements += batch['user'].batch_size
            data_loading_start = time.time()
        epoch_duration = time.time() - start_time
        computation_time = epoch_duration - total_data_loading_time
        print(f"Epoch {epoch}: Total Time = {epoch_duration}, Data Loading Time = {total_data_loading_time}, Computation Time = {computation_time}")
        model.lr_scheduler.step()
        timings.append(time.time() - start_time)

    print("avg_time = ", np.array(timings).mean())
