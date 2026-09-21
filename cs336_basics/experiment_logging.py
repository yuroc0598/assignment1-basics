import json
import os
import time
from pathlib import Path


class ExperimentLogger:
    """Records step/wall-clock/metric triples to a local JSONL file, optionally mirroring to wandb.

    The JSONL file is always written, so a run's history survives even if wandb
    is unavailable or `use_wandb=False`; wandb is purely an additional sink.
    """

    def __init__(
        self,
        run_name: str,
        config: dict,
        log_dir: str | os.PathLike = "runs",
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
    ):
        self.run_name = run_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{run_name}.jsonl"
        self._file = open(self.log_path, "w", encoding="utf-8")
        self._file.write(json.dumps({"type": "config", **config}) + "\n")
        self._file.flush()

        self._wandb_run = None
        if use_wandb:
            import wandb

            self._wandb_run = wandb.init(entity=wandb_entity, project=wandb_project, name=run_name, config=config)

        self.start_time = time.time()

    def log(self, step: int, **metrics: float) -> None:
        wall_clock = time.time() - self.start_time
        record = {"type": "metrics", "step": step, "wall_clock": wall_clock, **metrics}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        if self._wandb_run is not None:
            self._wandb_run.log({**metrics, "wall_clock": wall_clock}, step=step)

    def close(self) -> None:
        self._file.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
