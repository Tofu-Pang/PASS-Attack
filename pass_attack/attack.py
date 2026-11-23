from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.transforms import RandomResizedCrop


STEPS = 300
EPSILON = 16.0 / 255.0
INITIAL_STEP_SIZE = 2.0 / 255.0
ETA = 0.75
TAU = 0.01
STAGNATION_CHECKS = 2

RECORD_CHECKPOINTS = (45, 70, 95, 120, 140, 160, 180, 200)
ADAPTIVE_SCHEDULES = {
    "c1": (210, 220, 230, 240, 250, 260, 270, 280, 290, 300),
    "c2": (212, 224, 235, 246, 256, 266, 275, 284, 292, 300),
    "c3": (208, 216, 225, 234, 244, 254, 265, 276, 288, 300),
}
CHECKPOINTS = RECORD_CHECKPOINTS + ADAPTIVE_SCHEDULES["c2"]
CHECKPOINT_SCHEDULES = tuple(ADAPTIVE_SCHEDULES)

MOMENTUM_FACTOR = 1.0
INPUT_DIVERSITY_PROBABILITY = 0.7
TRANSLATION_KERNEL_SIZE = 5
TRANSLATION_KERNEL_SIGMA = 3.0
FEATURE_LAYER_DEPTH = 3
FEATURE_WEIGHT = 0.001
LOCAL_CROP_SCALE = 0.1


@dataclass(frozen=True)
class AttackSpec:
    name: str
    use_momentum: bool
    use_nesterov: bool
    use_su: bool
    use_pass: bool


ATTACK_SPECS = {
    "dtmi": AttackSpec("DTMI", True, False, False, False),
    "dtmi-pass": AttackSpec("DTMI-PASS", True, False, False, True),
    "dtmi-su": AttackSpec("DTMI-SU", True, False, True, False),
    "dtmi-su-pass": AttackSpec("DTMI-SU-PASS", True, False, True, True),
    "dtni": AttackSpec("DTNI", True, True, False, False),
    "dtni-pass": AttackSpec("DTNI-PASS", True, True, False, True),
    "dtni-su": AttackSpec("DTNI-SU", True, True, True, False),
    "dtni-su-pass": AttackSpec("DTNI-SU-PASS", True, True, True, True),
}
ATTACK_NAMES = tuple(ATTACK_SPECS)


def resolve_checkpoints(schedule="c2", steps=STEPS):
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be positive")
    key = str(schedule).lower()
    if key not in ADAPTIVE_SCHEDULES:
        raise ValueError(f"checkpoint_schedule must be one of {CHECKPOINT_SCHEDULES}")
    record = tuple(value for value in RECORD_CHECKPOINTS if value <= steps)
    adaptive = tuple(value for value in ADAPTIVE_SCHEDULES[key] if value <= steps)
    return record, adaptive


def classification_loss(logits, labels, loss_name, targeted):
    if loss_name == "ce":
        loss = F.cross_entropy(logits, labels, reduction="none")
        return loss if targeted else -loss
    if loss_name == "logit":
        selected = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        return -selected if targeted else selected
    raise ValueError("loss_name must be ce or logit")


class PASSScheduler:
    def __init__(
        self,
        batch_size,
        shape,
        initial_step_size,
        device,
        record_checkpoints,
        adaptive_checkpoints,
        eta=ETA,
        tau=TAU,
        stagnation_checks=STAGNATION_CHECKS,
        rollback=True,
    ):
        self.record_checkpoints = set(record_checkpoints)
        self.adaptive_checkpoints = set(adaptive_checkpoints)
        self.eta = float(eta)
        self.tau = float(tau)
        self.stagnation_checks = int(stagnation_checks)
        self.rollback = bool(rollback)
        self.step_size = torch.full((batch_size, 1, 1, 1), initial_step_size, device=device)
        self.best_loss = torch.full((batch_size,), float("inf"), device=device)
        self.best_delta = torch.zeros(shape, device=device)
        self.last_checkpoint_loss = self.best_loss.clone()
        self.stagnation_count = torch.zeros(batch_size, dtype=torch.int64, device=device)

    @torch.no_grad()
    def step(self, iteration, delta, eval_delta, loss):
        delta_base = delta
        improved = loss < self.best_loss
        self.best_loss = torch.where(improved, loss, self.best_loss)
        if improved.any():
            self.best_delta[improved] = eval_delta[improved]
        if iteration in self.record_checkpoints:
            self.last_checkpoint_loss = self.best_loss.clone()
            self.stagnation_count.zero_()
        elif iteration in self.adaptive_checkpoints:
            improvement = self.last_checkpoint_loss - self.best_loss
            stagnant = improvement <= self.tau
            self.stagnation_count = torch.where(
                stagnant,
                self.stagnation_count + 1,
                torch.zeros_like(self.stagnation_count),
            )
            triggered = self.stagnation_count >= self.stagnation_checks
            if triggered.any():
                self.step_size[triggered] *= self.eta
                if self.rollback:
                    delta_base = delta.clone()
                    delta_base[triggered] = self.best_delta[triggered]
                self.stagnation_count[triggered] = 0
            self.last_checkpoint_loss = self.best_loss.clone()
        return delta_base, self.step_size


class TransferAttack:
    def __init__(
        self,
        models,
        feature_layers,
        attack_name,
        loss_name,
        targeted,
        steps=STEPS,
        epsilon=EPSILON,
        initial_step_size=INITIAL_STEP_SIZE,
        eta=ETA,
        tau=TAU,
        checkpoint_schedule="c2",
        rollback=True,
    ):
        key = str(attack_name).lower()
        if key not in ATTACK_SPECS:
            raise ValueError(f"Unsupported attack: {attack_name}")
        self.spec = ATTACK_SPECS[key]
        self.models = list(models)
        if not self.models:
            raise ValueError("At least one source model is required")
        self.loss_name = str(loss_name).lower()
        self.targeted = bool(targeted)
        if self.loss_name not in ("ce", "logit"):
            raise ValueError("loss_name must be ce or logit")
        self.steps = int(steps)
        self.epsilon = float(epsilon)
        self.initial_step_size = float(initial_step_size)
        self.eta = float(eta)
        self.tau = float(tau)
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.epsilon <= 0.0 or self.initial_step_size <= 0.0:
            raise ValueError("epsilon and initial_step_size must be positive")
        if not 0.0 <= self.tau < float("inf"):
            raise ValueError("tau must be finite and nonnegative")
        if not 0.0 < self.eta < 1.0:
            raise ValueError("eta must be between zero and one")
        self.stagnation_checks = STAGNATION_CHECKS
        self.momentum_factor = MOMENTUM_FACTOR
        self.diversity_probability = INPUT_DIVERSITY_PROBABILITY
        self.feature_weight = FEATURE_WEIGHT
        self.rollback = bool(rollback)
        self.checkpoint_schedule = str(checkpoint_schedule).lower()
        self.record_checkpoints, self.adaptive_checkpoints = resolve_checkpoints(
            self.checkpoint_schedule,
            self.steps,
        )
        self.local_transform = RandomResizedCrop(
            299,
            scale=(LOCAL_CROP_SCALE, LOCAL_CROP_SCALE),
        )
        self.kernel = self._translation_kernel(
            TRANSLATION_KERNEL_SIZE,
            TRANSLATION_KERNEL_SIGMA,
        )
        self.activations = {}
        self.hooks = []
        if self.spec.use_su:
            if len(feature_layers) != len(self.models):
                raise ValueError("Each source model requires one feature layer")
            for index, layer in enumerate(feature_layers):
                self.hooks.append(layer.register_forward_hook(self._activation_hook(index)))

    def _activation_hook(self, index):
        def capture(module, inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            self.activations[index] = value

        return capture

    def _translation_kernel(self, size, sigma):
        if size < 1 or size % 2 == 0:
            raise ValueError("translation_kernel_size must be a positive odd number")
        coordinates = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        one_dimensional = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
        one_dimensional = one_dimensional / one_dimensional.sum()
        kernel = torch.outer(one_dimensional, one_dimensional)
        return kernel.expand(3, 1, size, size).contiguous()

    def _input_diversity(self, inputs):
        original = int(inputs.shape[-1])
        canvas = original + 31
        resized = int(torch.randint(original, canvas, (), device=inputs.device).item())
        top = int(torch.randint(0, canvas - resized + 1, (), device=inputs.device).item())
        left = int(torch.randint(0, canvas - resized + 1, (), device=inputs.device).item())
        bottom = canvas - resized - top
        right = canvas - resized - left
        if torch.rand((), device=inputs.device) > self.diversity_probability:
            return inputs
        resized_inputs = F.interpolate(inputs, size=(resized, resized), mode="nearest")
        return F.pad(resized_inputs, (left, right, top, bottom))

    def _normalize_gradient(self, gradient):
        denominator = gradient.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        return gradient / denominator

    def _source_loss(self, images, candidate_delta, labels):
        candidate = images + candidate_delta
        batch_size = candidate.shape[0]
        self.activations.clear()
        if self.spec.use_su:
            local = self.local_transform(images) + candidate_delta
            model_input = torch.cat((candidate, local), dim=0)
        else:
            model_input = candidate
        model_input = self._input_diversity(model_input)
        logits = torch.stack([model(model_input) for model in self.models], dim=0).mean(dim=0)
        if not self.spec.use_su:
            return classification_loss(logits, labels, self.loss_name, self.targeted)
        global_logits = logits[:batch_size]
        local_logits = logits[batch_size:]
        loss = classification_loss(global_logits, labels, self.loss_name, self.targeted)
        loss = loss + classification_loss(local_logits, labels, self.loss_name, self.targeted)
        similarities = []
        for index in range(len(self.models)):
            if index not in self.activations:
                raise RuntimeError("Feature activation was not captured")
            activation = self.activations[index]
            global_features = activation[:batch_size].reshape(batch_size, -1)
            local_features = activation[batch_size:].reshape(batch_size, -1)
            similarities.append(F.cosine_similarity(global_features, local_features, dim=1))
        similarity = torch.stack(similarities, dim=0).mean(dim=0)
        return loss - self.feature_weight * similarity

    def perturb(self, images, labels):
        device = images.device
        batch_size = images.shape[0]
        delta = torch.zeros_like(images)
        momentum = torch.zeros_like(images)
        step_size = torch.full((batch_size, 1, 1, 1), self.initial_step_size, device=device)
        scheduler = (
            PASSScheduler(
                batch_size=batch_size,
                shape=images.shape,
                initial_step_size=self.initial_step_size,
                device=device,
                record_checkpoints=self.record_checkpoints,
                adaptive_checkpoints=self.adaptive_checkpoints,
                eta=self.eta,
                tau=self.tau,
                stagnation_checks=self.stagnation_checks,
                rollback=self.rollback,
            )
            if self.spec.use_pass
            else None
        )
        kernel = self.kernel.to(device)
        padding = kernel.shape[-1] // 2

        for iteration in range(1, self.steps + 1):
            if self.spec.use_nesterov:
                candidate_delta = delta - self.momentum_factor * step_size * momentum
                candidate_delta = candidate_delta.clamp(-self.epsilon, self.epsilon)
            else:
                candidate_delta = delta
            candidate_delta = (images + candidate_delta).clamp(0.0, 1.0) - images
            candidate_delta = candidate_delta.detach().requires_grad_(True)

            sample_loss = self._source_loss(images, candidate_delta, labels)
            gradient = torch.autograd.grad(sample_loss.mean(), candidate_delta)[0]
            gradient = F.conv2d(gradient, kernel, padding=padding, groups=3)
            normalized = self._normalize_gradient(gradient)
            if self.spec.use_momentum:
                momentum = self.momentum_factor * momentum + normalized
            else:
                momentum = normalized

            with torch.no_grad():
                if scheduler is not None:
                    delta_base, step_size = scheduler.step(
                        iteration=iteration,
                        delta=delta,
                        eval_delta=candidate_delta.detach(),
                        loss=sample_loss.detach(),
                    )
                else:
                    delta_base = delta
                delta = delta_base - step_size * torch.sign(momentum)
                delta = delta.clamp(-self.epsilon, self.epsilon)
                delta = (images + delta).clamp(0.0, 1.0) - images

        return (images + delta).clamp(0.0, 1.0)

    def close(self):
        for hook in self.hooks:
            hook.remove()


def build_attack(
    models,
    feature_layers,
    attack_name,
    loss_name,
    targeted,
    steps=STEPS,
    epsilon=EPSILON,
    initial_step_size=INITIAL_STEP_SIZE,
    eta=ETA,
    tau=TAU,
    checkpoint_schedule="c2",
    rollback=True,
):
    return TransferAttack(
        models=models,
        feature_layers=feature_layers,
        attack_name=attack_name,
        loss_name=loss_name,
        targeted=targeted,
        steps=steps,
        epsilon=epsilon,
        initial_step_size=initial_step_size,
        eta=eta,
        tau=tau,
        checkpoint_schedule=checkpoint_schedule,
        rollback=rollback,
    )
