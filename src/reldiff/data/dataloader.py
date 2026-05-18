import torch
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput


def get_subgraph_dataloader(
    data: HeteroData,
    root_table: str | None = None,
    batch_size: int = 1,
    shuffle: bool = False,
    n_hops: int = 2,
    num_neighbors: int = -1,
    num_workers: int = 8,
    is_disjoint: bool = True,
    dimension_tables: list[str] = [],
    drop_last: bool = False,
    min_input_nodes: int = 0,
) -> NeighborLoader:
    r"""Return dataloaders for a given database, one dataloader for each table in the database."""
    num_neighbors = [num_neighbors] * n_hops
    if is_disjoint:
        assert root_table is not None, (
            "Root table must be provided for disjoint subgraph dataloader."
        )
        input_nodes = (root_table, None)
        indices = torch.arange(data[root_table].num_nodes)
        if shuffle and data[root_table].num_nodes < batch_size:
            # Repeat the indices to fill the batch size
            indices = torch.cat([indices] * (batch_size // data[root_table].num_nodes))
            input_nodes = (root_table, indices)
        return NeighborLoader(
            data.cpu(),
            num_neighbors=num_neighbors,
            input_nodes=input_nodes,
            batch_size=batch_size,
            subgraph_type="bidirectional",
            shuffle=shuffle,
            drop_last=False,
            replace=False,
            disjoint=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            input_id=indices,
        )
    else:
        return HeteroNeighborLoader(
            dataset=data.cpu(),
            batch_size=batch_size,
            n_hops=n_hops,
            replace=False,
            dimension_tables=dimension_tables,
            shuffle=shuffle,
            num_neighbors=num_neighbors,
            drop_last=drop_last,
            num_workers=num_workers,
            min_input_nodes=min_input_nodes,
        )


class HeteroNeighborLoader(DataLoader):
    def __init__(
        self,
        dataset: HeteroData,
        batch_size: int,
        n_hops: int = 2,
        replace: bool = False,
        dimension_tables: list = [],
        shuffle: bool = False,
        num_neighbors: int | list[int] = 64,
        drop_last: bool = False,
        num_workers: int = 0,
        min_input_nodes: int = 0,
    ):
        self.data = dataset
        self.replace = replace
        self.shuffle = shuffle
        self.n_hops = n_hops
        self.batch_size = batch_size
        self.min_input_nodes = min_input_nodes

        self.dimension_tables = dimension_tables
        if not isinstance(num_neighbors, list):
            num_neighbors = [num_neighbors] * n_hops

        self.num_neighbors = num_neighbors

        self.homogeneous = dataset.to_homogeneous(node_attrs=[], dummy_values=False)
        self.homogeneous.original_id = torch.zeros_like(self.homogeneous.node_type)

        self.dimension_mask = torch.zeros(self.homogeneous.num_nodes, dtype=torch.bool)
        self.node_ids = dict()
        for node_type_id, node_type in enumerate(dataset.node_types):
            mask = self.homogeneous.node_type == node_type_id
            self.homogeneous.original_id[mask] = torch.arange(mask.sum())
            if node_type in dimension_tables:
                self.dimension_mask[self.homogeneous.node_type == node_type_id] = True
            else:
                self.node_ids[node_type] = torch.where(mask)[0]

        self.neighbor_sampler = NeighborSampler(
            self.homogeneous,
            num_neighbors=self.num_neighbors,
            replace=replace,
            subgraph_type="induced",
            disjoint=False,
        )

        indices = torch.arange(self.data.num_nodes)
        # Remove dimension table indices
        indices = indices[~self.dimension_mask].numpy().tolist()
        super().__init__(
            indices,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    def sample_nodes(self, seed_ids):
        seed_data = NodeSamplerInput(
            input_id=seed_ids,
            node=seed_ids,
        )

        # Select the neighborhood of the seed nodes
        batch = self.neighbor_sampler.sample_from_nodes(seed_data)
        return batch.node

    def convert_to_heterodata(self, batch_ids, input_ids):
        node_ids = {
            table: torch.tensor([], dtype=torch.long) for table in self.data.node_types
        }
        input_nodes = {
            table: torch.tensor([], dtype=torch.long) for table in self.data.node_types
        }

        # Select input nodes
        input_mask = torch.isin(batch_ids, input_ids)
        for node_type_id, node_type in enumerate(self.data.node_types):
            # Select nodes of the same type
            node_type_mask = self.homogeneous.node_type[batch_ids] == node_type_id

            original_ids = self.homogeneous.original_id[batch_ids]

            node_ids[node_type] = original_ids[node_type_mask]
            input_nodes[node_type] = original_ids[node_type_mask & input_mask]

        subgraph = self.data.subgraph(node_ids)
        subgraph.set_value_dict("n_id", node_ids)
        subgraph.set_value_dict("input_id", input_nodes)
        return subgraph

    def collate_fn(self, index):
        input_ids = torch.tensor(index, dtype=torch.long)
        # a simple heuristic to ensure all model parameters are used
        if self.min_input_nodes > 0:
            # select min_input_nodes nodes from each table
            for node_ids in self.node_ids.values():
                # randomly select min_input_nodes from node_ids
                selected_ids = node_ids[
                    torch.randperm(len(node_ids))[: self.min_input_nodes]
                ]
                input_ids = torch.cat([input_ids, selected_ids], dim=0)
            input_ids = torch.unique(input_ids)
        node_ids = self.sample_nodes(input_ids)

        return self.convert_to_heterodata(node_ids, input_ids)
