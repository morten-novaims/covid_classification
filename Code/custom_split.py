"""Splits a folder with the given format:
    class1/
        img1.jpg
        img2.jpg
        ...
    class2/
        imgWhatever.jpg
        ...
    ...

into this resulting format:
    train/
        class1/
            img1.jpg
            ...
        class2/
            imga.jpg
            ...
    val/
        class1/
            img2.jpg
            ...
        class2/
            imgb.jpg
            ...
    test/
        class1/
            img3.jpg
            ...
        class2/
            imgc.jpg
            ...
"""

import pathlib
from pathlib import Path
import random
import shutil
import numpy as np
from os import path

try:
    from tqdm import tqdm

    tqdm_is_installed = True
except ImportError:
    tqdm_is_installed = False

# Set seed
seed = np.random.randint(100)


def list_dirs(directory):
    """Returns all directories in a given directory
    """
    return [f for f in Path(directory).iterdir() if f.is_dir()]


def list_dirs_classes(directory, classes):
    """Returns a list of paths of classes given they exists
    """
    return [Path(directory) / c for c in classes if (Path(directory) / c).is_dir()] #return [Path(directory + '/' + c) for c in classes if Path(directory + '/' + c).is_dir()]


def list_files(directory):
    """Returns all files in a given directory
    """
    return [f for f in Path(directory).iterdir() if f.is_file() and not f.name.startswith(".")]


def ratio(input, output="output", seed=seed, ratio=(0.8, 0.1, 0.1), classes=None):
    # make up for some impression
    assert round(sum(ratio), 5) == 1
    assert len(ratio) in (2, 3)

    if tqdm_is_installed:
        prog_bar = tqdm(desc=f"Copying files", unit=" files")

    if classes is None:
        dir_list = list_dirs(input)
    else:
        dir_list = list_dirs_classes(input, classes)
        assert len(dir_list) == len(classes)

    for class_dir in dir_list:
        split_class_dir_ratio(
            class_dir, output, ratio, seed, prog_bar if tqdm_is_installed else None
        )

    if tqdm_is_installed:
        prog_bar.close()


def fixed(input, output="output", seed=seed, fixed=(100, 100), oversample=False):
    # make sure its reproducible
    if isinstance(fixed, int):
        fixed = fixed

    assert len(fixed) in (1, 2)

    if tqdm_is_installed:
        prog_bar = tqdm(desc=f"Copying files", unit=" files")

    dirs = list_dirs(input)
    lens = []
    for class_dir in dirs:
        lens.append(
            split_class_dir_fixed(
                class_dir, output, fixed, seed, prog_bar if tqdm_is_installed else None
            )
        )

    if tqdm_is_installed:
        prog_bar.close()

    if not oversample:
        return

    max_len = max(lens)

    iteration = zip(lens, dirs)

    if tqdm_is_installed:
        iteration = tqdm(iteration, desc="Oversampling", unit=" classes")

    for length, class_dir in iteration:
        class_name = path.split(class_dir)[1]
        full_path = path.join(output, "train", class_name)
        train_files = list_files(full_path)
        for i in range(max_len - length):
            f_orig = random.choice(train_files)
            new_name = f_orig.stem + "_" + str(i) + f_orig.suffix
            f_dest = f_orig.with_name(new_name)
            shutil.copy2(f_orig, f_dest)


def setup_files(class_dir, seed):
    """Returns shuffled files
    """
    # make sure its reproducible
    random.seed(seed)

    files = list_files(class_dir)

    files.sort()
    random.shuffle(files)
    return files


def split_class_dir_fixed(class_dir, output, fixed, seed, prog_bar):
    """Splits one very class folder
    """
    files = setup_files(class_dir, seed)

    if not len(files) > sum(fixed):
        raise ValueError(
            f'The number of samples in class "{class_dir.stem}" are too few. There are only {len(files)} samples available but your fixed parameter {fixed} requires at least {sum(fixed)} files. You may want to split your classes by ratio.'
        )

    split_train = len(files) - sum(fixed)
    split_val = split_train + fixed[0]

    li = split_files(files, split_train, split_val, len(fixed) == 2)
    copy_files(li, class_dir, output, prog_bar)
    return len(files)


def split_class_dir_ratio(class_dir, output, ratio, seed, prog_bar):
    """Splits one very class folder
    """
    files = setup_files(class_dir, seed)

    split_train = int(ratio[0] * len(files))
    split_val = split_train + int(ratio[1] * len(files))

    li = split_files(files, split_train, split_val, len(ratio) == 3)
    copy_files(li, class_dir, output, prog_bar)


def split_files(files, split_train, split_val, use_test):
    """Splits the files along the provided indices
    """
    files_train = files[:split_train]
    files_val = files[split_train:split_val] if use_test else files[split_train:]

    li = [(files_train, "train"), (files_val, "val")]

    # optional test folder
    if use_test:
        files_test = files[split_val:]
        li.append((files_test, "test"))
    return li


def copy_files(files_type, class_dir, output, prog_bar):
    """Copies the files from the input folder to the output folder
    """
    # get the last part within the file
    class_name = path.split(class_dir)[1]
    for (files, folder_type) in files_type:
        full_path = path.join(output, folder_type, class_name)

        Path(full_path).mkdir(parents=True, exist_ok=True)
        for f in files:
            if not prog_bar is None:
                prog_bar.update()
            shutil.copy2(f, full_path)
