import matplotlib.pyplot as plt
import os
import pandas as pd


# 128 note on + 128 note off + 32 velocity + 100 timing
NUM_CHANNELS = 388

MAESTRO_DIRECTORY = os.getcwd() + '/data/maestro-v2.0.0'


def getTopComposers(df, num_composers):
  df['count'] = 1
  composer_counts = df\
    .groupby(by='canonical_composer')\
    .count()\
    .sort_values('count', ascending=False)['count']
  return composer_counts[:num_composers].keys()


def showComposerDistribution(df):
  df['count'] = 1
  df['train'] = df['split'] == 'train'
  df['validation'] = df['split'] == 'validation'
  df['test'] = df['split'] == 'test'
  composer_counts = df.groupby(by='canonical_composer')\
    .sum()\
    .sort_values('count', ascending=False)\
    .loc[:, ['train', 'validation', 'test']]

  composer_counts.plot.bar(stacked=True, figsize=(16, 5))
  plt.title('Distribution of composers in MAESTRO')
  plt.xlabel('')
  plt.ylabel('# of tracks')
  plt.show()


def getDataFrame():
  return pd.read_csv(F'{MAESTRO_DIRECTORY}/maestro-v2.0.0.csv')
