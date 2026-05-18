import argparse
import json
import os
import pickle
from glob import glob

import numpy as np
import torch
from syntherela.data import save_tables
from syntherela.metadata import Metadata

from reldiff.configs.utils import load_config, load_dataset_config
from reldiff.data import create_dataset, dataset_from_graph
from reldiff.data.utils import get_category_proportions
from reldiff.diffusion.unified_ctime_diffusion import \
    MultiTableUnifiedCtimeDiffusion
from reldiff.metrics import TabMetrics
from reldiff.models import GraphDiff, ModelJoint
from reldiff.sampler import MultiTableSampler

argparser = argparse.ArgumentParser()
argparser.add_argument("dataset_name", default="rossmann_subsampled", type=str)
argparser.add_argument("--num-timesteps", default=None, type=int)
argparser.add_argument(
    "--sampling-batch-size", default=20000, type=int, help="Batch size for sampling"
)
argparser.add_argument(
    "--structure",
    choices=["original", "generated", "2k"],
    default="original",
)
argparser.add_argument(
    "--config-path", default="src/reldiff/configs/reldiff_config.toml", type=str
)
argparser.add_argument("--dataset-config-path",default=None, type=str)
argparser.add_argument("--run-id", default="", type=str)
argparser.add_argument("--sampling-device", default="cuda", type=str)
argparser.add_argument(
    "--device", default="cuda", type=str, help="Device to use for inference"
)
argparser.add_argument(
    "--num-samples",
    default=1,
    type=int,
    help="Number of samples (synthetic databases) to generate",
)
argparser.add_argument("--use-ema", action="store_true", help="Use EMA for training")
argparser.add_argument("--compile-model", action="store_true", help="Use torch.compile")
args = argparser.parse_args()

database_name = args.dataset_name
structure = args.structure
run_id = args.run_id

# Load config
config = load_config(args.config_path)

if args.device == "cuda" and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if args.num_timesteps is not None:
    config["diffusion_params"]["num_timesteps"] = args.num_timesteps

settings = load_dataset_config(args.dataset_config_path)
is_disjoint = settings["is_disjoint"]
order_cols = settings["order_cols"]
dimension_tables = settings["dimension_tables"]
nh = settings["n_hops_dataloader"]
n_hops_dataloader = nh if nh is not None else config["graph"]["n_hops"]
if dimension_tables:
    print(
        f"Database {database_name} has the following dimension tables: {dimension_tables}"
    )
    config["diffusion_params"]["edm_params"]["net_conditioning"] = (
        "t"  # Condition on time as sigma is 0 for dimension tables
    )

DATA_DIR = "./data"
method_name = "RelDiff"

metadata_path = os.path.join(DATA_DIR, "original", database_name, "metadata.json")
metadata = Metadata().load_from_json(metadata_path)
data_path = os.path.join(DATA_DIR, "processed", database_name)
model_save_path = os.path.join("ckpt", database_name, "multi" + run_id)
result_save_path = os.path.join("results", database_name, "multi" + run_id)

dataset = create_dataset(
    metadata,
    data_path,
    order_cols=order_cols,
)

root_table = sorted(metadata.get_root_tables())[-1]
gnn_params = {
    "node_types": dataset.node_types,
    "edge_types": dataset.edge_types,
    "aggr": config["gnn"]["aggr"],
    "num_layers": config["graph"]["n_hops"],
    "type": config["gnn"]["type"],
}

order_dict = None
if set(metadata.get_tables()).intersection(set(order_cols.keys())):
    order_dict = dataset.order_dict


metrics_dict = dict()
categories = dict()
proportions_dict = dict()
for table in metadata.get_tables():
    # Skip foreign key only tables
    if table not in dataset.node_types:
        continue
    real_data_path = f"data/processed/{database_name}/{table}/data.csv"
    test_data_path = f"data/eval/{database_name}/{table}/data.csv"
    val_data_path = f"data/eval/{database_name}/{table}/data.csv"
    info_path = f"data/processed/{database_name}/{table}/info.json"
    with open(info_path, "r") as f:
        info = json.load(f)

    metrics_dict[table] = TabMetrics(
        real_data_path,
        test_data_path,
        val_data_path,
        info,
        device,
        metric_list=[
            "density",
            "c2st",
        ],
        drop_missing=True,
    )
    categories[table] = (np.array(dataset.categories_dict[table]) + 1).tolist()
    proportions_dict[table] = get_category_proportions(
        dataset[table].x_cat, categories[table], add_mask=True
    )

    if config["data"]["standardize"]:
        sigma_data = config["diffusion_params"]["edm_params"]["sigma_data"]
        # Check that the data is standardized
        assert np.isclose(dataset[table].x_num.mean(axis=0), 0, atol=1e-6).all()
        # Missing values can impact this
        assert np.isclose(dataset[table].x_num.std(axis=0), sigma_data, rtol=0.5).all()

    if is_disjoint:
        if table in dimension_tables:
            dataset[table].input_id = torch.tensor([]).long()
        else:
            # Set all nodes as input (target) nodes when using disjoint subgraphs.
            dataset[table].input_id = torch.arange(dataset[table].num_nodes).long()

if structure != "original":
    if structure == "generated":
        postfix = "_gen"
    else:
        postfix = f"_{structure}"
    # Load the generated graph
    with open(f"data/structure/{database_name}_graph{postfix}.pkl", "rb") as f:  #
        G = pickle.load(f)
    # Create the dataset directly from schema
    # do not add reverse edges as this is done in the dataset_from_graph function
    # and do not transform foreign key tables as this is done in the dataset_from_graph function
    dataset = create_dataset(
        metadata,
        data_path,
        order_cols=order_cols,
        add_reverse_edges=False,
        transform_fk_tables=False,
    )
    dataset = dataset_from_graph(
        G, dataset, metadata, dimension_tables=dimension_tables
    )
    for table in dataset.node_types:
        # Set all nodes as input (target) nodes when using disjoint subgraphs.
        dataset[table].input_id = torch.arange(dataset[table].num_nodes).long()
else:
    postfix = ""

zero_init = config["data"]["standardize"]
backbone = GraphDiff(
    dataset.d_numerical_dict,
    categories,  # dataset.categories_dict + 1 for mask state
    gnn_params,
    **config["model"],
    zero_init=zero_init,
    proportions_dict=proportions_dict,
    order_enc=order_dict,
).cuda()


model = ModelJoint(
    denoise_fn=backbone,
    **config["diffusion_params"]["edm_params"],
)

if args.compile_model:
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    model = torch.compile(model)

assert n_hops_dataloader == config["graph"]["n_hops"] or is_disjoint

diffusion = MultiTableUnifiedCtimeDiffusion(
    num_classes=dataset.categories_dict,
    num_numerical_features=dataset.d_numerical_dict,
    denoise_fn=model,
    **config["diffusion_params"],
    device=device,
    root_table=root_table,
    n_hops_dataloader=n_hops_dataloader,
    dequantize=config["data"]["dequantize"],
    dataset=dataset,
    is_disjoint=is_disjoint,
    num_neighbors=config["graph"]["num_neighbors"],
    dimension_tables=dimension_tables,
)


print(f"Loading model from {model_save_path}")
checkpoint_files = glob(os.path.join(model_save_path, "best_model*"))

latest_checkpoint = checkpoint_files[0]
diffusion.load_state_dict(torch.load(latest_checkpoint, weights_only=True))
print(f"Loaded checkpoint from {latest_checkpoint}")

num_params = sum(p.numel() for p in diffusion.parameters())
print("The number of parameters = ", num_params)
diffusion.to(device)
diffusion.eval()

sampler = MultiTableSampler(
    diffusion,
    dataset,
    metrics_dict=metrics_dict,
    device=device,
    dimension_tables=dimension_tables,
    sampling_device=args.sampling_device,
    sampling_batch_size=args.sampling_batch_size,
)

for i in range(args.num_samples):
    print(f"Sampling synthetic tables for sample {i + 1}...")
    synthetic_tables = sampler.sample_database(
        database_name,
        metadata,
        dataset,
        mask_missing=True,
        batch_size=args.sampling_batch_size,
    )

    save_tables(
        synthetic_tables,
        f"data/synthetic/{database_name}/{method_name}{postfix}/{run_id}/sample{i + 1}",
    )

# EMA model
if args.use_ema:
    print(f"Loading EMA model from {model_save_path}")
    checkpoint_files = glob(os.path.join(model_save_path, "best_ema_model*"))

    latest_checkpoint = checkpoint_files[0]
    diffusion.load_state_dict(torch.load(latest_checkpoint, weights_only=True))
    print(f"Loaded EMA checkpoint from {latest_checkpoint}")
    diffusion.to(device)
    diffusion.eval()

    sampler = MultiTableSampler(
        diffusion,
        dataset,
        metrics_dict=metrics_dict,
        device=device,
        dimension_tables=dimension_tables,
        sampling_device=args.sampling_device,
        sampling_batch_size=args.sampling_batch_size,
    )

    for i in range(args.num_samples):
        print(f"Sampling synthetic tables for sample {i + 1} with EMA model...")
        synthetic_tables = sampler.sample_database(
            database_name,
            metadata,
            dataset,
            mask_missing=True,
            batch_size=args.sampling_batch_size,
        )

        save_tables(
            synthetic_tables,
            f"data/synthetic/{database_name}/{method_name}{postfix}_EMA/{run_id}/sample{i + 1}",
        )
