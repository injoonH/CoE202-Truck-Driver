import pandas as pd

if __name__ == '__main__':
    FILE = 'Road2_GoalDist5(20211029_22-52-42).txt'

    df = pd.read_csv(f'../Logs/{FILE}').sort_values('frame', ascending=False)
    df_goal = df[df.goal == 1].sort_values('frame')

    print('All :', df.shape[0])
    print('Goal:', df_goal.shape[0])

    print(df_goal.head())
