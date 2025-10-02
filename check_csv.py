import pandas as pd

df = pd.read_csv("fairface_eval.csv")

print("Label counts:")
print(df['label'].value_counts())
print("\nGroup counts:")
print(df['group'].value_counts())
