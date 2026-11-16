from .config import (
    DATASETS,
    DEVICE,
    SEEDS,
    RUN_ALL_SEEDS,
)

from .data_loader import (
    load_dataset,
    set_seed,
    configure_cpu,
    dataset_info,
)

from .edge_encoder import (
    create_static_edge_attributes,
)

from .models import (
    build_model,
)

from .train import (
    train_model,
)

from .evaluate import (
    save_results,
)


MODELS = (
    "GAT",
    "GATv2",
    "GATv2 + Static Edge",
    "GATv2 + Learned Edge",
)


def run_single_seed(
    seed,
):

    print()
    print("=" * 80)
    print(
        f"RUNNING SEED: {seed}"
    )
    print("=" * 80)

    set_seed(
        seed
    )

    results = []

    for dataset_name in DATASETS:

        print()
        print("=" * 80)
        print(
            f"DATASET: {dataset_name}"
        )
        print("=" * 80)

        dataset, data = (
            load_dataset(
                dataset_name
            )
        )

        info = dataset_info(
            dataset,
            data
        )

        print(
            f"Nodes      : "
            f"{info['nodes']}"
        )

        print(
            f"Edges      : "
            f"{info['edges']}"
        )

        print(
            f"Features   : "
            f"{info['features']}"
        )

        print(
            f"Classes    : "
            f"{info['classes']}"
        )

        print(
            f"Train      : "
            f"{info['train_nodes']}"
        )

        print(
            f"Validation : "
            f"{info['val_nodes']}"
        )

        print(
            f"Test       : "
            f"{info['test_nodes']}"
        )

        # ----------------------------------------------------
        # Static edge features
        # ----------------------------------------------------

        static_edge_attr = (
            create_static_edge_attributes(
                data.x,
                data.edge_index,
                edge_dim=16,
            )
        )

        for model_name in MODELS:

            print()
            print("-" * 80)

            print(
                f"{dataset_name} | "
                f"{model_name}"
            )

            print("-" * 80)

            # Same seed for every model.
            set_seed(
                seed
            )

            model = build_model(
                model_name=
                    model_name,

                in_channels=
                    dataset.num_features,

                num_classes=
                    dataset.num_classes,
            )

            edge_attr = None

            if (
                model_name ==
                "GATv2 + Static Edge"
            ):

                edge_attr = (
                    static_edge_attr
                )

            parameters = sum(
                p.numel()
                for p in
                model.parameters()
            )

            print(
                f"Parameters : "
                f"{parameters:,}"
            )

            metrics = train_model(
                model=model,
                data=data,
                device=DEVICE,
                edge_attr=edge_attr,
            )

            result = {

                "Seed":
                    seed,

                "Dataset":
                    dataset_name,

                "Model":
                    model_name,

                **metrics,
            }

            results.append(
                result
            )

            print()

            print(
                f"Best validation accuracy : "
                f"{metrics['Validation Accuracy']:.4f}"
            )

            print(
                f"Best validation Macro F1  : "
                f"{metrics['Validation Macro F1']:.4f}"
            )

            print(
                f"Final test accuracy       : "
                f"{metrics['Test Accuracy']:.4f}"
            )

            print(
                f"Test Macro F1              : "
                f"{metrics['Macro F1']:.4f}"
            )

            print(
                f"Parameters                : "
                f"{metrics['Parameters']:,}"
            )

            print(
                f"Best epoch                : "
                f"{metrics['Best Epoch']}"
            )

            print(
                f"Training time             : "
                f"{metrics['Training Time']:.2f}s"
            )

    return results


def main():

    configure_cpu()

    print("=" * 80)
    print(
        "EDGE-ENHANCED GRAPH ATTENTION NETWORKS"
    )
    print("=" * 80)

    print(
        f"Device : {DEVICE}"
    )

    print(
        f"Models : "
        f"{', '.join(MODELS)}"
    )

    if RUN_ALL_SEEDS:

        seeds_to_run = SEEDS

    else:

        seeds_to_run = (
            SEEDS[:1]
        )

    all_results = []

    for seed in (
        seeds_to_run
    ):

        seed_results = (
            run_single_seed(
                seed
            )
        )

        all_results.extend(
            seed_results
        )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print("=" * 100)
    print(
        "FINAL RESEARCH RESULTS"
    )
    print("=" * 100)

    print(
        f"{'Dataset':<12}"
        f"{'Model':<32}"
        f"{'Val F1':>9}"
        f"{'Test':>9}"
        f"{'Test F1':>9}"
        f"{'Params':>12}"
    )

    print("-" * 100)

    for result in (
        all_results
    ):

        print(
            f"{result['Dataset']:<12}"
            f"{result['Model']:<32}"
            f"{result['Validation Macro F1']:>9.4f}"
            f"{result['Test Accuracy']:>9.4f}"
            f"{result['Macro F1']:>9.4f}"
            f"{result['Parameters']:>12,}"
        )

    save_results(
        all_results
    )

    print()
    print("=" * 80)
    print(
        "EXPERIMENT COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()