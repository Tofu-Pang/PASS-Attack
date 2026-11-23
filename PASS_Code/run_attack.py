import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from pass_attack import (
    ATTACK_NAMES,
    ATTACK_SPECS,
    CHECKPOINT_SCHEDULES,
    EPSILON,
    ETA,
    FEATURE_LAYER_DEPTH,
    INITIAL_STEP_SIZE,
    MODEL_NAMES,
    NIPSDataset,
    STEPS,
    TAU,
    build_attack,
    build_model,
    resolve_checkpoints,
    resolve_feature_layer,
)


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"model_groups", "model_labels", "attacks", "default"}
    if not required.issubset(config):
        raise ValueError(f"Missing configuration keys: {sorted(required - set(config))}")
    required_groups = {"standard", "diverse"}
    if not required_groups.issubset(config["model_groups"]):
        raise ValueError(f"Missing model groups: {sorted(required_groups - set(config['model_groups']))}")
    for names in config["model_groups"].values():
        invalid = sorted(set(names) - set(MODEL_NAMES))
        if invalid:
            raise ValueError(f"Unsupported models in configuration: {invalid}")
    if set(config["model_labels"]) != set(MODEL_NAMES):
        raise ValueError("model_labels must cover all supported models")
    if set(config["attacks"]) != set(ATTACK_NAMES):
        raise ValueError("attacks must match the supported attack list")
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Run PASS transfer adversarial attack.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data", default="data")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--attack", choices=ATTACK_NAMES, default="dtni-su-pass")
    parser.add_argument("--loss", choices=("ce", "logit"), default="ce")
    parser.add_argument("--mode", choices=("targeted", "untargeted"), default="targeted")
    parser.add_argument("--source-models", nargs="+")
    parser.add_argument("--target-models", nargs="+")
    parser.add_argument("--source-group")
    parser.add_argument("--target-group")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--epsilon", type=float, default=EPSILON * 255.0)
    parser.add_argument("--alpha", type=float, default=INITIAL_STEP_SIZE * 255.0)
    parser.add_argument("--eta", type=float, default=ETA)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument(
        "--checkpoint-schedule",
        choices=CHECKPOINT_SCHEDULES,
        default="c2",
    )
    parser.add_argument("--no-rollback", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def resolve_names(args, config):
    groups = config["model_groups"]
    if args.source_models and args.source_group:
        raise ValueError("Choose source-models or source-group")
    if args.target_models and args.target_group:
        raise ValueError("Choose target-models or target-group")
    if args.source_group:
        if args.source_group not in groups:
            raise ValueError(f"Unknown source group: {args.source_group}")
        source_names = list(groups[args.source_group])
    elif args.source_models:
        source_names = list(args.source_models)
    else:
        source_names = list(config["default"]["source_models"])
    if args.target_group:
        if args.target_group not in groups:
            raise ValueError(f"Unknown target group: {args.target_group}")
        target_names = list(groups[args.target_group])
    elif args.target_models:
        target_names = list(args.target_models)
    else:
        target_names = list(config["default"]["target_models"])
    invalid = sorted((set(source_names) | set(target_names)) - set(MODEL_NAMES))
    if invalid:
        raise ValueError(f"Unsupported models: {invalid}")
    overlap = sorted(set(source_names) & set(target_names))
    if overlap:
        raise ValueError(f"Source and target models overlap: {overlap}")
    if not source_names or not target_names:
        raise ValueError("Source and target model lists must be nonempty")
    return source_names, target_names


def execute(args):
    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    data_path = Path(args.data)
    if not config_path.is_absolute():
        config_path = root / config_path
    if not data_path.is_absolute():
        data_path = root / data_path
    config = load_config(config_path)
    source_names, target_names = resolve_names(args, config)
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.epsilon <= 0.0 or args.alpha <= 0.0:
        raise ValueError("epsilon and alpha must be positive")
    if ATTACK_SPECS[args.attack].use_pass:
        if not 0.0 <= args.tau < float("inf"):
            raise ValueError("tau must be finite and nonnegative")
        if not 0.0 < args.eta < 1.0:
            raise ValueError("eta must be between zero and one")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    dataset = NIPSDataset(data_path)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("max-samples must be positive")
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    source_models = [build_model(name, device) for name in source_names]
    target_models = [build_model(name, device) for name in target_names]
    use_su = ATTACK_SPECS[args.attack].use_su
    feature_layers = (
        [
            resolve_feature_layer(model, name, FEATURE_LAYER_DEPTH)
            for model, name in zip(source_models, source_names)
        ]
        if use_su
        else []
    )
    targeted = args.mode == "targeted"
    attack = build_attack(
        models=source_models,
        feature_layers=feature_layers,
        attack_name=args.attack,
        loss_name=args.loss,
        targeted=targeted,
        steps=args.steps,
        epsilon=args.epsilon / 255.0,
        initial_step_size=args.alpha / 255.0,
        eta=args.eta,
        tau=args.tau,
        checkpoint_schedule=args.checkpoint_schedule,
        rollback=not args.no_rollback,
    )
    successes = torch.zeros(len(target_models), dtype=torch.int64)
    total = 0
    label = ATTACK_SPECS[args.attack].name
    try:
        for images, true_labels, target_labels in tqdm(loader, desc=label, unit="batch"):
            images = images.to(device, non_blocking=True)
            labels = target_labels if targeted else true_labels
            labels = labels.to(device, non_blocking=True)
            adversarial = attack.perturb(images, labels)
            with torch.no_grad():
                for index, model in enumerate(target_models):
                    predictions = model(adversarial).argmax(dim=1)
                    success = predictions == labels if targeted else predictions != labels
                    successes[index] += success.sum().cpu()
            total += images.shape[0]
    finally:
        attack.close()
    result = {
        "attack": label,
        "mode": args.mode,
        "loss": args.loss,
        "source_models": source_names,
        "target_models": target_names,
        "samples": total,
        "asr_percent": {
            name: 100.0 * int(successes[index]) / total
            for index, name in enumerate(target_names)
        },
    }
    if ATTACK_SPECS[args.attack].use_pass:
        record_cps, adaptive_cps = resolve_checkpoints(args.checkpoint_schedule, args.steps)
        result["pass"] = {
            "checkpoint_schedule": args.checkpoint_schedule,
            "record_checkpoints": list(record_cps),
            "adaptive_checkpoints": list(adaptive_cps),
            "rollback": not args.no_rollback,
            "eta": args.eta,
            "tau": args.tau,
        }
    return result


def main():
    result = execute(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
