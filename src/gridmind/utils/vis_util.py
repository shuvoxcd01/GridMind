from typing import Dict, Hashable, Optional
from matplotlib import pyplot as plt
from tabulate import tabulate
import numpy as np
import os
import glob
import logging


def print_state_action_values(
    q_table: Dict[Hashable, np.ndarray], filename: str = None
):
    single_key = list(q_table.keys())[0]
    num_actions = len(q_table[single_key])
    dict_to_print = {"A\u2193S\u2192 ": [i for i in range(num_actions)]}
    q_table_sorted = dict(sorted(q_table.items()))
    dict_to_print.update(q_table_sorted)

    if filename is not None:
        with open(filename, "w") as file:
            print(tabulate(dict_to_print, headers="keys", tablefmt="grid"), file=file)
    else:
        print(tabulate(dict_to_print, headers="keys", tablefmt="grid"))


def plot_state_values(states, true_values, estimated_values):
    """
    Plots the true values and estimated values for each state.

    :param states: List of state names (e.g., ['A', 'B', 'C', ...])
    :param true_values: List of true values corresponding to the states
    :param estimated_values: List of lists containing estimated values for each state over iterations
    """

    # Create a figure and axis
    plt.figure(figsize=(10, 6))

    # Plot the true values (these will be constant across all iterations)
    plt.plot(
        states,
        true_values,
        label="True Values",
        color="black",
        linestyle="--",
        marker="o",
    )

    # Plot estimated values over iterations
    for i, estimate in enumerate(estimated_values):
        plt.plot(states, estimate, label=f"Iteration {i+1}", linestyle="-", marker="x")

    # Adding labels and title
    plt.xlabel("States")
    plt.ylabel("Values")
    plt.title("True vs Estimated Values of States")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.tight_layout()
    plt.savefig("True vs Estimated Values of States.png")
    plt.show()


def print_value_table(
    feature1,
    feature2,
    state_values,
    feature1_name="Feature1",
    feature2_name="Feature2",
    filename: str = None,
    append: bool = False,
):
    # Ensure inputs are of the same length
    if not (len(feature1) == len(feature2) == len(state_values)):
        raise ValueError("All input lists must have the same length.")

    # Create a sorted list of unique values for each feature
    feature1_values = sorted(set(feature1))
    feature2_values = sorted(set(feature2))

    # Create a mapping from (feature1, feature2) to state_values
    value_dict = {
        (f1, f2): "{:.2f}".format(val)
        for f1, f2, val in zip(feature1, feature2, state_values)
    }

    # Build the table grid
    table = []
    for f1 in feature1_values:
        row = [f1]  # Start with the row header
        for f2 in feature2_values:
            row.append(value_dict.get((f1, f2), "N/A"))  # Use "N/A" if no value exists
        table.append(row)

    # Add column headers and print the table
    headers = [f"{feature1_name} \\ {feature2_name}"] + feature2_values

    if filename is not None:
        filemode = "a" if append else "w"
        with open(filename, mode=filemode) as file:
            print(tabulate(table, headers=headers, tablefmt="grid"), file=file)
            file.write("\n")

    else:
        print(tabulate(table, headers=headers, tablefmt="grid"))
        print("\n")


class VideoUtil:
    """Utility class for video loading and processing operations."""

    @staticmethod
    def load_video_as_tensor(
        video_save_path: str, logger: Optional[logging.Logger] = None
    ):
        """
        Load video file(s) and convert to tensor format for TensorBoard.

        Args:
            video_save_path: Base path for the video files (without extension)
            logger: Optional logger for logging messages

        Returns:
            torch.Tensor: Video tensor in format (N, T, C, H, W) where:
                N = batch size (1)
                T = number of frames
                C = channels (3 for RGB)
                H = height
                W = width
            Returns None if video cannot be loaded.
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        try:
            import torch
            import torchvision
            from torchvision.io import read_video
            import gc
        except ImportError:
            logger.warning(
                "torch or torchvision not installed. Cannot load video for TensorBoard. "
                "Install with: pip install torch torchvision"
            )
            return None

        # RecordVideo creates files with pattern: {name_prefix}-episode-{id}.mp4
        # Find the most recent video file matching the pattern
        video_dir = os.path.dirname(video_save_path)
        video_prefix = os.path.basename(video_save_path)

        # Search for video files with the prefix
        video_files = glob.glob(os.path.join(video_dir, f"{video_prefix}*.mp4"))

        if not video_files:
            logger.warning(
                f"No video files found matching pattern: {video_save_path}*.mp4"
            )
            return None

        # Use the most recently created video file
        latest_video = max(video_files, key=os.path.getctime)
        logger.info(f"Loading video from: {latest_video}")

        video_frames = None
        try:
            # read_video returns (video_tensor, audio_tensor, info)
            # video_tensor shape: (T, H, W, C)
            video_tensor, audio_tensor, info = read_video(latest_video, pts_unit="sec")

            # Explicitly delete audio tensor to free memory
            del audio_tensor

            # Convert from (T, H, W, C) to (N, T, C, H, W) format expected by TensorBoard
            # Permute dimensions: T, H, W, C -> T, C, H, W
            video_frames = video_tensor.permute(0, 3, 1, 2)

            # Delete original tensor to free memory
            del video_tensor

            # Add batch dimension: T, C, H, W -> 1, T, C, H, W
            video_frames = video_frames.unsqueeze(0)

            logger.info(f"Loaded video with shape: {video_frames.shape}")

            # Force garbage collection to free up memory
            gc.collect()

            return video_frames

        except Exception as e:
            logger.error(f"Error loading video {latest_video}: {str(e)}")
            # Clean up on error
            if video_frames is not None:
                del video_frames
            gc.collect()
            return None


if __name__ == "__main__":
    # Example usage
    feature1 = [0, 0, 1, 1]
    feature2 = [0, 1, 0, 1]
    state_values = [1.0, 0.5, 0.8, 0.2]

    print_value_table(
        feature1, feature2, state_values, feature1_name="X-axis", feature2_name="Y-axis"
    )
