import pandas as pd
import os

if __name__ == '__main__':
    log_path = '../Logs/'
    files = os.listdir(log_path)
    df = pd.concat([pd.read_csv(log_path + file) for file in files])
    df = df.drop_duplicates().sort_values('frame', ascending=False)
    df_goal = df[df.goal == 1].sort_values('frame')

    print('All :', df.shape[0])
    print('Goal:', df_goal.shape[0])

    print(df_goal.head())
