from exp.shared.runner import train_one


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    train_one(args.dataset, "forecast_targeted_penalty", args.seed)
