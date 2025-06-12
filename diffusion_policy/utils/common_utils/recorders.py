import atexit
import glob
import os
import signal
from enum import Enum
from typing import Dict, Optional, Union

import cv2
import imageio
import numpy as np


class ActMode(Enum):
    # NOTE: Waypoint used to be 1, so prior datasets may be inconsisten
    # Should be ok for this project since we are using dense all the way through
    Waypoint = 0
    Dense = 1
    Terminate = 2
    Interpolate = 3


class DatasetRecorder:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        self._reset()

    def _reset(self):
        self.episode = []
        self.images = []
        self.waypoint_idx = -1

    def record(
        self,
        mode: ActMode,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        click_pos: Optional[np.ndarray] = None,
    ):
        """mode: delta, position, waypoint"""
        if mode == ActMode.Waypoint:
            print("Recording Click:", action)
            self.waypoint_idx += 1
            waypoint_idx = self.waypoint_idx
        elif mode == ActMode.Dense:
            # print("Recording Dense:", action)
            waypoint_idx = -1
        elif mode == ActMode.Interpolate:
            print("Recording Interpolate:", action)
            waypoint_idx = self.waypoint_idx

        self.episode.append(
            {
                "obs": obs,
                "action": action,
                "mode": mode,
                # "waypoint_idx": waypoint_idx,
                "click": click_pos,
            }
        )

        views = []
        for k, v in obs.items():
            if "image" in k and v.ndim == 3:
                views.append(cv2.resize(v, (320, 240)))
        self.images.append(np.hstack(views))

    def end_episode(self, save, save_gif=False, additional_info=None):
        if save:
            existing_demos = glob.glob(os.path.join(self.data_folder, "demo*.npz"))
            if len(existing_demos) == 0:
                next_idx = 0
            else:
                existing_indices = [
                    int(fname.split("/")[-1].split(".")[0][len("demo") :])
                    for fname in existing_demos
                ]
                next_idx = np.max(existing_indices) + 1

            demo_path = os.path.join(self.data_folder, f"demo%05d.npz" % next_idx)

            data = {"episode": self.episode}
            if additional_info is not None:
                data.update(**additional_info)
            np.savez(demo_path, **data)
            print(f"Demo saved to {demo_path}")

            if save_gif:
                gif_path = os.path.join(self.data_folder, f"demo%05d.gif" % next_idx)
                print(f"saving to {gif_path}")
                for i in range(len(self.images)):
                    vis = self.images[i]
                    # vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    if self.episode[i]["mode"] == ActMode.Dense:
                        vis[:10, :, :] = (0, 255, 0)
                    self.images[i] = vis
                    cv2.imshow("vis", vis)
                    cv2.waitKey(20)
                imageio.mimsave(gif_path, self.images, duration=0.03, loop=0)
        else:
            print("Episode discarded")

        self._reset()


class StreamRecorder:
    def __init__(self, data_folder, additional_info={}):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        self._reset(additional_info)
        self._setup_graceful_exit()

    def _reset(self, additional_info={}):
        # from termcolor import cprint
        self.episode = []
        self.images = []
        self.additional_info = additional_info

    def _setup_graceful_exit(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        atexit.register(self._save_on_exit)

    def _signal_handler(self, sig, frame):
        print("Ctrl+C detected. Saving data and exiting...")
        self._save_on_exit()
        exit(0)

    def _save_on_exit(self):
        if self.episode:
            self.end_episode(save=True, save_gif=True)

    def record(
        self,
        obs: Dict[str, np.ndarray],
        gif_image: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    ):
        self.episode.append(obs)

        if gif_image is FileNotFoundError:
            return

        views = []
        if isinstance(gif_image, np.ndarray):
            views.append(gif_image)
        elif isinstance(gif_image, dict):
            for k, v in gif_image.items():
                if "image" in k and v.ndim == 3:
                    views.append(v)
        self.images.append(np.hstack(views))

    def end_episode(self, save=True, save_gif=True):
        if save:
            existing_demos = glob.glob(os.path.join(self.data_folder, "demo*.npz"))
            next_idx = len(existing_demos)
            demo_path = os.path.join(self.data_folder, f"demo{next_idx:05d}.npz")
            np.savez(
                demo_path,
                episode=self.episode,
                additional_info=self.additional_info,
                allow_pickle=False,
            )
            print(f"Demo saved to {demo_path}")
            if save_gif and self.images:
                self._save_gif(next_idx)
        self._reset()

    def _save_gif(self, idx):
        gif_path = os.path.join(self.data_folder, f"demo{idx:05d}.gif")
        print(f"Saving GIF to {gif_path}")
        imageio.mimsave(gif_path, self.images, duration=0.03, loop=0)


class PointCloudRecorder(DatasetRecorder):
    def record(self, obs, pointcloud, action):
        self.episode.append(
            dict(
                pointcloud=pointcloud,
                action=action,
                mode=ActMode.Dense,
            )
        )
        views = []
        for k, v in obs.items():
            if "image" in k and v.ndim == 3:
                views.append(cv2.resize(v, (320, 240)))
        self.images.append(np.hstack(views))
