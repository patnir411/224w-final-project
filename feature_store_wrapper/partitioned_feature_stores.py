import numpy as np
from feature_store_wrapper.cached_feature_store import PartitionCachedFeatureStore
from feature_store_wrapper.cluster_utils import PartitionData, compute_partition_data


class PartitionedFeatureStoreIterable(object):
    r"""
    Generate iterable of PartitionCachedFeatureStore to be used in node-wise
    sampling.

    Args:
      node_indices: Input node indices with which to perform node-wise sampling

      cached_attrs: Node features which will be cached in-memory/on GPU

      feature_store: Underlying on-disk feature store

      edge_index: 2xn tensor corresponding to all the edges in the graph
          (assumes edge relationship type is ignored for heterogeneous graphs)

      num_parts: Number of partitions

    Returns:
      PartitionCachedFeatureStore, node_indices: FeatureStore and input node
          indices to be provided to node-wise sampler (e.g. NeighborSampler)

    """
    def __init__(self, node_indices, cached_attrs, feature_store, edge_index, num_parts):
        partition_data = compute_partition_data(edge_index, num_parts)
        self.cluster_idx = partition_data.node_mapping[node_indices]
        self.num_parts = len(partition_data.partptr) - 1
        self.partptr = partition_data.partptr
        self.node_mapping = partition_data.node_mapping
        self.base_feature_store = feature_store
        self.cached_attrs = cached_attrs
        self.curr_iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_iter < self.num_parts:
            lo, hi = self.partptr[self.curr_iter], self.partptr[self.curr_iter+1]
            feature_store = self._get_feature_store((lo, hi))
            idxs = self.cluster_idx[np.logical_and(self.cluster_idx >= lo, self.cluster_idx <= hi)]
            self.curr_iter += 1
            return feature_store, idxs
        raise StopIteration

    def _get_feature_store(self, bound):
      feature_map = {}
      for attr in self.cached_attrs:
        feature_map[attr] = self.base_feature_store[attr, bound[0]:bound[1]]
      return PartitionCachedFeatureStore(self.node_mapping, feature_map, bound[0], bound[1], self.base_feature_store)
