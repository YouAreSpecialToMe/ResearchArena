from exp.shared.stressaudit import PRIMARY_SEED, freeze_inventory, run_method_experiment, split_inventory, tune_threshold


if __name__ == "__main__":
    inventory = split_inventory(freeze_inventory())
    threshold = tune_threshold("lexical", inventory)
    run_method_experiment("lexical", inventory, threshold, run_seed=PRIMARY_SEED, repeat_tag="headline")
