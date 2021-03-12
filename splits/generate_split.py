import random
import pandas as pd
from pathlib import Path


def generate(fpath, split_name, train_prop=.9):
    path = Path(fpath).resolve()
    print(f"Loading files from {path}")
    files = [i.stem for i in path.iterdir()]
    # must omit very first and very last frames, because loads i-1 and i+1
    files.remove("{:010d}".format(len(files)))
    files.remove("{:010d}".format(1))
    #print("Start frame:", min(files))
    #print("End frame:", max(files))

    save_dir = Path(__file__).resolve().parent.joinpath(split_name)
    save_dir.mkdir(exist_ok=True)

    random.shuffle(files)
    split_index = int(train_prop * len(files))
    trainset = files[:split_index]
    valset = files[split_index:]
    print("Dataset size:", len(files))
    print("Trainset size:", len(trainset))
    print("Valset size:", len(valset))

    train_df = pd.DataFrame(trainset, columns=["stem"])
    val_df = pd.DataFrame(valset, columns=["stem"])
    for df in (train_df, val_df):
        df["root"] = "./"
        df["camera"] = "c"

    col_save_order = ["root", "stem", "camera"]
    train_df = train_df[col_save_order]
    val_df = val_df[col_save_order]
    train_df.to_csv(save_dir.joinpath("train_files.txt"), sep=' ', header=None, index=None, mode='w')
    val_df.to_csv(save_dir.joinpath("val_files.txt"), sep=' ', header=None, index=None, mode='w')
    print(f"Saved train_files.txt and val_files.txt to {save_dir}")

if __name__ == "__main__":
    generate(
        fpath = "/data/mattbev/traces/blue_prius_cambridge_rain/frames/camera_front", #_00.01.00_to_00.02.00",
        split_name = "blue_prius_cambridge_rain", #_00.01.00_to_00.02.00",
        train_prop = 0.9
    )
