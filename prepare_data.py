import torch
import kuzu
import numpy as np
import pandas as pd

from os import path
from tqdm import tqdm


def get_twibot22_db(is_populated=False):
    DATABASE_NAME = "twibot22/TwiBot22-DB"
    DATASET_PATH = "twibot22"

    # print("extracting users")
    # user = pd.read_json(path.join(DATASET_PATH, 'user.json'))
    # user_idx = user['id']
    # uid_index = {uid:index for index,uid in enumerate(user_idx.values)}

    # print("extracting edge_index")
    # edge = pd.read_csv(path.join(DATASET_PATH, 'edge.csv'))
    # follows_edge_index = []
    # following_edge_index = []

    # for i in tqdm(range(len(edge))):
    #     sid=edge['source_id'][i]
    #     tid=edge['target_id'][i]
    #     if edge['relation'][i]=='followers':
    #         try:
    #             follows_edge_index.append([uid_index[sid],uid_index[tid]])
    #         except KeyError:
    #             continue
    #     elif edge['relation'][i]=='following':
    #         try:
    #             following_edge_index.append([uid_index[sid],uid_index[tid]])
    #         except KeyError:
    #             continue

    FOLLOWS_EDGE_INDEX_PATH = path.join(DATASET_PATH, "follows_edge_index.csv")
    FOLLOWING_EDGE_INDEX_PATH = path.join(DATASET_PATH, "following_edge_index.csv")

    follows_edge_index = pd.read_csv(FOLLOWS_EDGE_INDEX_PATH).to_numpy()
    following_edge_index = pd.read_csv(FOLLOWING_EDGE_INDEX_PATH).to_numpy()

    # follows_edge_index = np.array(follows_edge_index)
    # following_edge_index = np.array(following_edge_index)

    # follows_edge_index_df = pd.DataFrame(follows_edge_index)
    # follows_edge_index_df.to_csv(FOLLOWS_EDGE_INDEX_PATH, index=False)

    # following_edge_index_df = pd.DataFrame(following_edge_index)
    # following_edge_index_df.to_csv(FOLLOWING_EDGE_INDEX_PATH, index=False)

    num_nodes = max(follows_edge_index.max(), following_edge_index.max()) + 1

    ID_PATH = path.join(DATASET_PATH, "node_id.npy")
    CAT_FEATURE_PATH = path.join(DATASET_PATH, "cat_feature.npy")
    PROP_FEATURE_PATH = path.join(DATASET_PATH, "num_properties_feature.npy")
    TWEET_FEATURE_PATH = path.join(DATASET_PATH, "tweet_feature.npy")
    DES_FEATURE_PATH = path.join(DATASET_PATH, "des_feature.npy")
    LABEL_PATH = path.join(DATASET_PATH, "label.npy")

    # Generate random feature vectors due to memory issue with data processing
    cat_features = np.random.rand(num_nodes, 3).astype("float32")
    prop_features = np.random.rand(num_nodes, 5).astype("float32")
    tweet_features = np.random.rand(num_nodes, 768).astype("float32")
    des_feature = np.random.rand(num_nodes, 768).astype("float32")
    labels = np.random.randint(2, size=num_nodes)

    np.save(ID_PATH, np.arange(num_nodes))
    np.save(CAT_FEATURE_PATH, cat_features)
    np.save(PROP_FEATURE_PATH, prop_features)
    np.save(TWEET_FEATURE_PATH, tweet_features)
    np.save(DES_FEATURE_PATH, des_feature)
    np.save(LABEL_PATH, labels)

    db = kuzu.Database(DATABASE_NAME)
    conn = kuzu.Connection(db, num_threads=1)

    print("Creating features table")
    conn.execute(
        "CREATE NODE TABLE user(id INT64, cat_features FLOAT[3], prop_features FLOAT[5], tweet_features FLOAT[768], des_features FLOAT[768], y INT64, "
        "PRIMARY KEY (id));"
    )
    conn.execute(
        'COPY user FROM ("%s",  "%s",  "%s", "%s", "%s", "%s") BY COLUMN'
        % (
            ID_PATH,
            CAT_FEATURE_PATH,
            PROP_FEATURE_PATH,
            TWEET_FEATURE_PATH,
            DES_FEATURE_PATH,
            LABEL_PATH,
        )
    )

    print("Creating edges table")
    conn.execute("CREATE REL TABLE follows(FROM user TO user, MANY_MANY);")
    conn.execute("CREATE REL TABLE following(FROM user TO user, MANY_MANY);")
    conn.execute('COPY follows FROM "%s" (HEADER=true)' % (FOLLOWS_EDGE_INDEX_PATH))
    conn.execute('COPY following FROM "%s" (HEADER=true)' % (FOLLOWING_EDGE_INDEX_PATH))
    return db


def get_twibot20_db(is_populated=False):
    DATABASE_NAME = "twibot20/TwiBot20-DB"
    db = kuzu.Database(DATABASE_NAME)
    if is_populated:
        return db

    DATASET_PATH = "twibot20/processed_data"
    cat_features = torch.load(path.join(DATASET_PATH, "cat_properties_tensor.pt"))
    prop_features = torch.load(path.join(DATASET_PATH, "num_properties_tensor.pt"))
    tweet_features = torch.load(path.join(DATASET_PATH, "tweets_tensor.pt"))
    des_features = torch.load(path.join(DATASET_PATH, "des_tensor.pt"))

    labels = torch.load(path.join(DATASET_PATH, "label.pt"))
    num_nodes = cat_features.shape[0]
    expanded_labels = np.full(num_nodes, -100)
    expanded_labels[: len(labels)] = labels
    node_ids = np.arange(num_nodes)

    ID_PATH = path.join(DATASET_PATH, "node_id.npy")
    CAT_FEATURE_PATH = path.join(DATASET_PATH, "cat_feature.npy")
    PROP_FEATURE_PATH = path.join(DATASET_PATH, "num_properties_feature.npy")
    TWEET_FEATURE_PATH = path.join(DATASET_PATH, "tweet_feature.npy")
    DES_FEATURE_PATH = path.join(DATASET_PATH, "des_feature.npy")
    LABEL_PATH = path.join(DATASET_PATH, "label.npy")

    np.save(CAT_FEATURE_PATH, cat_features.numpy())
    np.save(PROP_FEATURE_PATH, prop_features.numpy())
    np.save(TWEET_FEATURE_PATH, tweet_features.numpy())
    np.save(DES_FEATURE_PATH, des_features.numpy())
    np.save(LABEL_PATH, expanded_labels)
    np.save(ID_PATH, node_ids)

    FOLLOWS_EDGE_INDEX_PATH = path.join(DATASET_PATH, "follows_edge_index.csv")
    FOLLOWING_EDGE_INDEX_PATH = path.join(DATASET_PATH, "following_edge_index.csv")
    edge_index = torch.load(path.join(DATASET_PATH, "edge_index.pt"))
    edge_type = torch.load(path.join(DATASET_PATH, "edge_type.pt"))
    follows_edge_index = edge_index[:, edge_type == 0].T.numpy()
    follows_edge_index_df = pd.DataFrame(follows_edge_index)
    follows_edge_index_df.to_csv(FOLLOWS_EDGE_INDEX_PATH, index=False)
    following_edge_index = edge_index[:, edge_type == 1].T.numpy()
    following_edge_index_df = pd.DataFrame(following_edge_index)
    following_edge_index_df.to_csv(FOLLOWING_EDGE_INDEX_PATH, index=False)

    db = kuzu.Database(DATABASE_NAME)
    conn = kuzu.Connection(db, num_threads=1)

    print("Creating features table")
    conn.execute(
        "CREATE NODE TABLE user(id INT64, cat_features FLOAT[3], prop_features FLOAT[5], tweet_features FLOAT[768], des_features FLOAT[768], y INT64, "
        "PRIMARY KEY (id));"
    )
    conn.execute(
        'COPY user FROM ("%s",  "%s",  "%s", "%s", "%s", "%s") BY COLUMN'
        % (
            ID_PATH,
            CAT_FEATURE_PATH,
            PROP_FEATURE_PATH,
            TWEET_FEATURE_PATH,
            DES_FEATURE_PATH,
            LABEL_PATH,
        )
    )

    print("Creating edges table")
    conn.execute("CREATE REL TABLE follows(FROM user TO user, MANY_MANY);")
    conn.execute("CREATE REL TABLE following(FROM user TO user, MANY_MANY);")
    conn.execute('COPY follows FROM "%s" (HEADER=true)' % (FOLLOWS_EDGE_INDEX_PATH))
    conn.execute('COPY following FROM "%s" (HEADER=true)' % (FOLLOWING_EDGE_INDEX_PATH))
    return db


if __name__ == "__main__":
    # db = get_twibot20_db(is_populated=False)
    # feature_store, graph_store = db.get_torch_geometric_remote_backend(num_threads=1)

    # from torch_geometric.loader import NeighborLoader
    # train_idx = torch.arange(0, 8278)
    # loader = NeighborLoader(data=(feature_store, graph_store), num_neighbors={('user', 'follows', 'user'): [6, 6, 6], ('user', 'following', 'user'): [6,6,6]}, input_nodes=('user', train_idx), batch_size=128, shuffle=True, filter_per_worker=False)
    # for batch in loader:
    #     print(batch['user'].batch_size)
    get_twibot22_db(is_populated=False)
