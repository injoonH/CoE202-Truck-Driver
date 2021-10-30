import pandas as pd
import os

if __name__ == '__main__':
    LOG_PATH = '../Logs/Road2/'

    files = os.listdir(LOG_PATH)
    df = pd.concat([pd.read_csv(LOG_PATH + file) for file in files])
    df = df.drop_duplicates().sort_values('frame', ascending=False)
    df.reset_index(drop=True, inplace=True)
    df_goal = df[df.goal == 1].sort_values('frame')
    df_goal.reset_index(drop=True, inplace=True)

    print('All :', df.shape[0])
    print('Goal:', df_goal.shape[0])

    print(df_goal.head(10))
