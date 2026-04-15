from exp.shared.runner import train_one


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--relaxloss-lambda", type=float, default=None)
    args = parser.parse_args()
    train_one(args.dataset, "relaxloss", args.seed, relaxloss_lambda=args.relaxloss_lambda)
