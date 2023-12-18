import torch
import numpy as np
from torch_geometric.data.feature_store import FeatureStore, TensorAttr
from torch_geometric.typing import FeatureTensorType
from typing import List, Optional, Tuple
from kuzu.types import Type


class PartitionCachedFeatureStore(FeatureStore):
    r"""A wrapper around any disk-access based FeatureStore which caches the features of nodes
    in a graph partition (where part_lower_bound <= node_id < part_upper_bound) either
    in-memory or on GPU. 

    Args:
        cluster_node_id_map (np.array): Mapping from node_id in original graph
            to node_id in relabelled graph after partitioning
        features_map (dict): Dictionary mapping graph attribute to their corresponding features
            for nodes in the partition
        part_lower_bound (int):
        part_upper_bound (int):
        base_feature_store (FeatureStore): Underlying FeatureStore object that supports
            fetching from disk
    """

    def __init__(
        self, cluster_node_id_map, features_map, part_lower_bound, part_upper_bound, base_feature_store
    ):
        super().__init__()
        self.cluster_node_id_map = cluster_node_id_map
        self.features_map = features_map
        self.part_lower_bound = part_lower_bound
        self.part_upper_bound = part_upper_bound
        self.base_feature_store = base_feature_store

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        table_name = attr.group_name
        attr_name = attr.attr_name

        if (table_name, attr_name) in self.features_map:
            cached_features = self.features_map[table_name, attr_name]
            indices = self.cluster_node_id_map[attr.index]
            cached_indices = (
                np.logical_and(indices >= self.part_lower_bound, indices < self.part_upper_bound)
            ).nonzero()
            non_cached_indices = (
                np.logical_or(indices < self.part_lower_bound, indices >= self.part_upper_bound)
            ).nonzero()
            if len(cached_indices[0]) == 0:
                return self.base_feature_store._get_tensor(attr)
            elif len(non_cached_indices[0]) == 0:
                return cached_features[indices]
            else:
                cached_result = cached_features[indices[cached_indices]]
                attr.index = np.array(attr.index)[non_cached_indices]
                non_cached_result = self.base_feature_store._get_tensor(attr)
                feature_dim = cached_result.shape[1]
                result = torch.empty((len(indices), feature_dim))
                result[cached_indices] = cached_result
                result[non_cached_indices] = non_cached_result
                return result

        return self.base_feature_store._get_tensor(attr)

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        raise self.base_feature_store._remove_tensor()

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple:
        return self.base_feature_store._get_tensor_size(attr)

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        return self.base_feature_store.get_all_tensor_attrs()
