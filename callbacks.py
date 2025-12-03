from stable_baselines3.common.callbacks import BaseCallback
import csv
import os


class EpisodeOutcomeCallback(BaseCallback):
    """
    Tracks how many episodes end in lifted / knocked_down / timeout.

    Assumes env.step(...)'s info dict has boolean keys:
      info["lifted"], info["knocked_down"], info["timeout"]
    and that these are meaningful on terminal steps (terminated or truncated).

    Works with VecEnv.
    """

    def __init__(self, log_every_episodes: int = 100, csv_path: str | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.log_every_episodes = log_every_episodes
        self.csv_path = csv_path

        self.n_episodes = 0
        self.n_lifted = 0
        self.n_knocked_down = 0
        self.n_timeout = 0

        self._csv_file = None
        self._csv_writer = None

    def _on_training_start(self) -> None:
        # optional CSV logging
        if self.csv_path is not None:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "total_timesteps",
                "episodes",
                "lifted",
                "knocked_down",
                "timeout",
                "lifted_rate",
                "knocked_rate",
                "timeout_rate",
            ])

    def _on_step(self) -> bool:
        # In SB3, for VecEnv callbacks, we get "dones" and "infos" in self.locals
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is None or infos is None:
            return True

        for env_idx, done in enumerate(dones):
            if not done:
                continue

            info = infos[env_idx] or {}

            lifted = bool(info.get("lifted", False))
            knocked = bool(info.get("knocked_down", False))
            timeout = bool(info.get("timeout", False))

            # Increment counters
            self.n_episodes += 1
            if lifted:
                self.n_lifted += 1
            elif knocked:
                self.n_knocked_down += 1
            else:
                # treat "neither lifted nor knocked" as timeout
                self.n_timeout += 1

            # Log every N episodes
            if self.n_episodes % self.log_every_episodes == 0:
                lifted_rate = self.n_lifted / self.n_episodes
                knocked_rate = self.n_knocked_down / self.n_episodes
                timeout_rate = self.n_timeout / self.n_episodes

                # Log to TensorBoard
                self.logger.record("custom/episodes", float(self.n_episodes))
                self.logger.record("custom/success_rate", lifted_rate)
                self.logger.record("custom/knockdown_rate", knocked_rate)
                self.logger.record("custom/timeout_rate", timeout_rate)

                if self.verbose > 0:
                    print(
                        f"[Outcome] steps={self.num_timesteps} "
                        f"episodes={self.n_episodes} | "
                        f"lifted={lifted_rate:.3f}, "
                        f"knocked={knocked_rate:.3f}, "
                        f"timeout={timeout_rate:.3f}"
                    )

                # Also write to CSV if enabled
                if self._csv_writer is not None:
                    self._csv_writer.writerow([
                        self.num_timesteps,
                        self.n_episodes,
                        self.n_lifted,
                        self.n_knocked_down,
                        self.n_timeout,
                        lifted_rate,
                        knocked_rate,
                        timeout_rate,
                    ])

        return True

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
