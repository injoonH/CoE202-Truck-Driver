import pandas as pd

df = pd.read_csv('../Logs/Road2_GoalDist5(20211029_22-52-42).txt').sort_values('frame', ascending=False)
df_goal = df[df.goal == 1].sort_values('frame')

print('All :', df.shape[0])
print('Goal:', df_goal.shape[0])

print(df_goal.head())
