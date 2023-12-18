from torch_geometric.data.feature_store import FeatureStore, TensorAttr
from torch_geometric.typing import FeatureTensorType
from typing import List, Optional, Tuple
from kuzu.types import Type
import numpy as np

import pyg_lib
import numpy as np
from torch_geometric.utils import index_sort, sort_edge_index
from torch_geometric.utils.sparse import index2ptr
from dataclasses import dataclass


@dataclass
class PartitionData(object):
    r"""
    Graph partitioning is represented as follows:
    * Nodes from range partptr[i] to partptr[i+1] is in the i^{th} partition
    * Node IDs are permuted after partitioning. In particular, node_perm[i]
      denotes the original node ID of the i^{th} node after partion relabelling,
      while node_mapping[j] denote the partition-relabelled node ID of the
      j^{th} node in the original graph
    """
    node_perm: np.array
    partptr: np.array
    node_mapping: np.array


def compute_partition_data(
    edge_index, num_parts, node_perm_path=None, partptr_path=None
):
    r"""Clusters/partitions a graph into multiple partitions via :obj:`METIS`
    algorithm, as described in `"A Software Package for Partitioning
    Unstructured Graphs, Partitioning Meshes, and Computing Fill-Reducing
    Orderings of Sparse Matrices"` _ paper.

    Args:
        edge_index (torch.Tensor): Compressed source node indices.
        num_partitions (int): The number of partitions.
        node_perm_path (string, optional): File to save node's permutation after
            partitioning
        partptr_path (string, optional): File to save node id range of each
            partition

    Returns:
        PartionData: Node partitioning after running METIS algorithm
    """
    num_nodes = edge_index.max().item() + 1
    row, index = sort_edge_index(edge_index, num_nodes)
    indptr = index2ptr(row, size=num_nodes)

    cluster = pyg_lib.partition.metis(
        indptr.cpu(),
        index.cpu(),
        num_partitions=num_parts,
        recursive=False,
    )

    cluster, node_perm = index_sort(cluster, max_value=num_parts)
    partptr = index2ptr(cluster, size=num_parts)

    if node_perm_path is not None:
        np.save(node_perm_path, node_perm)
        np.save(partptr_path, partptr)

    node_mapping = np.zeros(len(node_perm))
    node_mapping[node_perm] = np.arange(len(node_perm), dtype="int64")
    return PartitionData(node_perm.numpy(), partptr.numpy(), node_mapping)


def load_partition_data(node_perm_file, partptr_file):
    node_perm = np.load(node_perm_file)
    partptr = np.load(partptr_file)
    node_mapping = np.zeros(len(node_perm))
    node_mapping[node_perm] = np.arange(len(node_perm), dtype="int64")
    return PartitionData(node_perm, partptr, node_mapping)
