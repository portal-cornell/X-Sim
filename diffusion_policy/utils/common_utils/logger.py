import json
import time
import numpy as np
import sys
import imageio

try:
    import wandb
except ImportError:
    pass


class Logger:
    def __init__(
        self,
        log_dir,
        step,
        print_to_stdout=True,
        use_tensorboard=False,
        use_wandb=False,
        wandb_kwargs={},
    ):
        self._logdir = log_dir
        self._logdir.mkdir(parents=True, exist_ok=True)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._gifs = {}
        self.step = step

        # JSONL logging setup
        self._jsonl_file = open(self._logdir / "metrics.jsonl", "a")

        # Console and file logging setup
        if print_to_stdout:
            self.terminal = sys.stdout
        else:
            self.terminal = None

        # Create directories for media files
        self._image_dir = self._logdir / "images"
        self._video_dir = self._logdir / "videos"
        self._video_dir.mkdir(exist_ok=True)
        self._gif_dir = self._logdir / "gifs"
        self._gif_dir.mkdir(exist_ok=True)

        # TensorBoard setup (optional)
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(
                log_dir=f"{str(log_dir)}/tensorboard", max_queue=1000
            )

        # Wandb setup (optional)
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.require("core")
            wandb.init(**wandb_kwargs)

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def gif(self, name, value, fps=10):
        self._gifs[name] = {"frames": np.array(value), "fps": fps}

    def write(self, message):
        if self.terminal is not None:
            self.terminal.write(message)

    def flush(self):
        if self.terminal is not None:
            self.terminal.flush()
        self._jsonl_file.flush()

    def log_metrics(self, fps=False, step=False):
        if not step:
            step = self.step
        scalars = self._scalars.copy()
        if fps:
            scalars["fps"] = self._compute_fps(step)

        log_message = (
            f"[{step}] " + " / ".join(f"{k} {v:.1f}" for k, v in scalars.items()) + "\n"
        )
        self.write(log_message)

        # Log to JSONL
        self._log_to_jsonl(step, scalars)

        # Log to TensorBoard (if enabled)
        if self.use_tensorboard:
            self._log_to_tensorboard(step, scalars)

        # Log to Wandb (if enabled)
        if self.use_wandb:
            self._log_to_wandb(step, scalars)

        # Save images, videos, and GIFs locally
        self._save_media_files(step)

        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._gifs = {}

    def _save_media_files(self, step):
        for name, image in self._images.items():
            self._image_dir.mkdir(exist_ok=True)
            image_path = self._image_dir / f"{name}_{step}.png"
            imageio.imwrite(image_path, image)

        for name, video in self._videos.items():
            self._video_dir.mkdir(exist_ok=True)
            video_path = self._video_dir / f"{name}_{step}.mp4"
            imageio.mimsave(video_path, video, fps=30)

        for name, gif_data in self._gifs.items():
            self._gif_dir.mkdir(exist_ok=True)
            gif_path = self._gif_dir / f"{name}_{step}.gif"
            imageio.mimsave(gif_path, gif_data["frames"], fps=gif_data["fps"])

    def _log_to_jsonl(self, step, scalars):
        log_data = {"step": step, **scalars}
        self._jsonl_file.write(json.dumps(log_data) + "\n")
        self._jsonl_file.flush()

    def _log_to_tensorboard(self, step, scalars):
        for name, value in scalars.items():
            self._writer.add_scalar(name, value, step)
        for name, image in self._images.items():
            self._writer.add_image(name, image, step)
        for name, video in self._videos.items():
            self._writer.add_video(name, video.unsqueeze(0), step, fps=30)

    def _log_to_wandb(self, step, scalars):
        log_dict = {**scalars, "step": step}
        for name, image in self._images.items():
            log_dict[name] = wandb.Image(image)
        for name, video in self._videos.items():
            log_dict[name] = wandb.Video(video)
        for name, gif_data in self._gifs.items():
            # (B, T, H, W, C) -> (B, T, C, H, W)
            wandb_gif = np.moveaxis(gif_data["frames"], -1, -3)
            log_dict[name] = wandb.Video(wandb_gif, fps=gif_data["fps"], format="gif")
        wandb.log(log_dict)

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        log_data = {"step": step, name: value}
        self._jsonl_file.write(json.dumps(log_data) + "\n")
        self._jsonl_file.flush()
        if self.use_tensorboard:
            self._writer.add_scalar(name, value, step)
        if self.use_wandb:
            wandb.log({name: value, "step": step})

    def offline_video(self, name, value, step):
        video_path = self._video_dir / f"{name}_{step}.mp4"
        imageio.mimsave(video_path, value, fps=30)
        if self.use_tensorboard:
            self._writer.add_video(name, value.unsqueeze(0), step, fps=30)
        if self.use_wandb:
            wandb.log({name: wandb.Video(value), "step": step})

    def offline_gif(self, name, value, step, fps=10):
        gif_path = self._gif_dir / f"{name}_{step}.gif"
        imageio.mimsave(gif_path, value, fps=fps)
        if self.use_wandb:
            wandb.log({name: wandb.Video(value, fps=fps, format="gif"), "step": step})

    def close(self):
        self._jsonl_file.close()
        if self.use_tensorboard:
            self._writer.close()
        if self.use_wandb:
            wandb.finish()

    def __del__(self):
        self.close()
