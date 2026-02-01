import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset("titanic")
print(df.loc[1:3,["age","class"]])
print(df.columns)
sns.violinplot(x=df["age"])
plt.show()