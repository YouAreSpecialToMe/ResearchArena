from exp.shared.stressaudit import freeze_inventory, run_data_inventory_experiment, split_inventory


if __name__ == "__main__":
    run_data_inventory_experiment(split_inventory(freeze_inventory()))
