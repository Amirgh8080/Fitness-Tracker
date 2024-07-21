import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv(
    '../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv')

single_file_grc = pd.read_csv(
    '../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv')

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob('../../data/raw/MetaMotion/*.csv')
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

data_path = "../../data/raw/MetaMotion\\"
f = files[0]

paticipant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip('123').rstrip("_MetaWear_2019")

df = pd.read_csv(f)
df['paticipant'] = paticipant
df['label'] = label
df['category'] = category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
grc_df = pd.DataFrame()

acc_set = 1
grc_set = 1

for f in files:
    paticipant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip('123').rstrip("_MetaWear_2019")

    df = pd.read_csv(f)
    df['paticipant'] = paticipant
    df['label'] = label
    df['category'] = category

    if "Gyroscope" in f:
        df['set'] = grc_set
        grc_set += 1
        grc_df = pd.concat([grc_df, df])
    if "Accelerometer" in f:
        df['set'] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.info()

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit='ms')
grc_df.index = pd.to_datetime(grc_df["epoch (ms)"], unit='ms')

del acc_df['epoch (ms)']
del acc_df['time (01:00)']
del acc_df['elapsed (s)']

del grc_df['epoch (ms)']
del grc_df['time (01:00)']
del grc_df['elapsed (s)']


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob('../../data/raw/MetaMotion/*.csv')


def read_data_from_files(files):
    acc_df = pd.DataFrame()
    grc_df = pd.DataFrame()

    acc_set = 1
    grc_set = 1

    for f in files:
        paticipant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip('123').rstrip("_MetaWear_2019")

        df = pd.read_csv(f)
        df['paticipant'] = paticipant
        df['label'] = label
        df['category'] = category

        if "Gyroscope" in f:
            df['set'] = grc_set
            grc_set += 1
            grc_df = pd.concat([grc_df, df])
        if "Accelerometer" in f:
            df['set'] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit='ms')
    grc_df.index = pd.to_datetime(grc_df["epoch (ms)"], unit='ms')

    del acc_df['epoch (ms)']
    del acc_df['time (01:00)']
    del acc_df['elapsed (s)']

    del grc_df['epoch (ms)']
    del grc_df['time (01:00)']
    del grc_df['elapsed (s)']

    return acc_df, grc_df


acc_df, grc_df = read_data_from_files(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

data_merged = pd.concat([acc_df.iloc[:, :3], grc_df], axis=1)

data_merged.columns = [
    'acc_x',
    'acc_y',
    'acc_z',
    'gyr_x',
    'gyr_y',
    'gyr_z',
    'participant',
    'label',
    'category',
    'set'
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    'acc_x': "mean",
    'acc_y': "mean",
    'acc_z': "mean",
    'gyr_x': "mean",
    'gyr_y': "mean",
    'gyr_z': "mean",
    'label': "last",
    'category': "last",
    'participant': "last",
    'set': "last",
}

data_merged.resample(rule='200ms').apply(sampling)

# split by days
days = [g for n, g in data_merged.groupby(pd.Grouper(freq='D'))]

data_resampled = pd.concat(
    [df.resample(rule='200ms').apply(sampling).dropna() for df in days])

data_resampled.info()

data_resampled['set'] = data_resampled['set'].astype('int')


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle('../../data/interim/01_data_processed.pkl')
