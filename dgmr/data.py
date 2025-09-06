import datetime as dt
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
from scipy.ndimage import zoom

from dgmr.settings import DATA_PATH, INPUT_STEPS, TIMESTEP, RADAR_FILE_DATE_FORMAT


def get_files_list(date: dt.datetime) -> List[Path]:
    delta = dt.timedelta(minutes=TIMESTEP)
    dates = [date + i * delta for i in range(-INPUT_STEPS + 1, 1)]
    filenames = [d.strftime(RADAR_FILE_DATE_FORMAT) for d in dates]
    return [DATA_PATH / f for f in filenames]


def open_radar_file(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as ds:
        array = np.array(ds["dataset1"]["data1"]["data"])
    return array


def get_input_array(paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to open h5 radar files and get the radar field mask

    Args:
        paths: List of paths to h5 radar files
    Returns:
        Input array of shape (timesteps, x_size, y_size)
        Input mask of shape (x_size, y_size)
    """
    arrays = [open_radar_file(path) for path in paths]

    # Put values outside radar field to 0
    mask = np.where(arrays[0] == 65535, 1, 0)
    arrays = [np.where(array == 65535, 0, array) for array in arrays]

    # Rescale to 1km resolution
    arrays = [zoom(array, (0.5, 0.5)) for array in arrays]
    mask = zoom(mask, (0.5, 0.5))

    array = np.stack(arrays)
    array = array / 100 * 12  # Conversion from mm cumulated in 5min to mm/h
    array = np.expand_dims(array, -1)  # Add channel dims
    return array, mask
