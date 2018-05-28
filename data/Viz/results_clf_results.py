import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# BASE
clf_names = ["Naive Bayes" for _ in range(1)] + ["CART" for _ in range(1)] + ["Random Forest" for _ in range(1)] + \
            ["Multilayer Perc" for _ in range(1)] + ["DNN 2 Layer" for _ in range(1)] + ["DNN 3 Layer" for _ in range(1)]
df_names = ["Blank" for _ in range(6)] + ["Pre-Game" for _ in range(6)] + ["In-Game" for _ in range(6)] + \
        ["Inventory" for _ in range(6)] + ["Performance" for _ in range(6)] + ["Patch Context" for _ in range(6)]
variable = ['AVG', 'STD']

# item_base
avg_items_base = [
    19.64,
    22.41,
    22.78,
    14.62,
    22.72,
    22.71,

    19.50,
    24.83,
    23.82,
    22.06,
    23.57,
    23.61,

    19.55,
    25.03,
    24.97,
    24.02,
    24.24,
    24.51,

    19.56,
    31.93,
    36.93,
    36.98,
    46.51,
    46.71,

    19.71,
    36.90,
    40.12,
    40.25,
    53.77,
    53.34,

    0.87,
    37.04,
    38.37,
    38.33,
    54.27,
    54.54,
]
std_items_base = [
    0.54,
    0.7,
    0.73,
    0.69,
    0.18,
    0.4,

    0.59,
    0.98,
    0.97,
    1.78,
    0.61,
    0.62,

    0.66,
    0.88,
    0.61,
    0.96,
    0.46,
    0.39,

    0.6,
    0.91,
    1.43,
    2.27,
    1.00,
    0.67,

    0.58,
    0.93,
    1.31,
    3.61,
    1.01,
    0.73,

    0.2,
    0.87,
    1.22,
    2.22,
    1.10,
    0.91
]
df_items_base = pd.DataFrame({
    'clf': clf_names*6,
    'data': df_names,
    'avg': avg_items_base,
    'std': std_items_base
    },
    columns=['clf', 'data', 'avg', 'std']
)
df_items_base.to_csv("df_items_base.csv", sep=";")

a4_dims = (9, 5)
fig, ax = plt.subplots(figsize=a4_dims)
g = sns.barplot(x='data', y='avg', hue='clf', data=df_items_base, palette='Blues')
g.set(xlabel='Dataset', ylabel='Mean Precision in %')
ax.set(ylim=(0, 100))
ax.yaxis.set_ticks(np.arange(0, 100, 10))
ax.legend(loc='upper left', #bbox_to_anchor=(0.5, 1.05),
          fancybox=True, shadow=True, ncol=2)
plt.setp(ax.get_legend().get_texts(), fontsize='8')
fig.tight_layout()
fig.savefig("items_base.svg")
plt.clf()


# skills_base
avg_skills_base = [
    42.67,
    51.02,
    51.66,
    42.19,
    42.96,
    60.03,

    42.22,
    60.03,
    58.86,
    47.42,
    43.28,
    44.69,

    42.14,
    88.34,
    87.88,
    81.67,
    61.62,
    58.69,

    43.38,
    88.03,
    76.88,
    73.74,
    61.58,
    62.49,

    44.33,
    88.02,
    80.21,
    81.72,
    65.81,
    63.61,

    41.87,
    88.16,
    78.01,
    67.33,
    64.77,
    59.68,
]
std_skills_base = [
    0.92,
    1.44,
    1.66,
    0.77,
    1.93,
    1.58,
    1.13,
    1.58,
    1.49,
    1.08,
    2.58,
    1.79,
    1.07,
    0.88,
    0.40,
    7.15,
    5.47,
    3.31,
    1.07,
    0.87,
    1.41,
    6.62,
    7.20,
    5.59,
    1.57,
    0.64,
    2.14,
    4.28,
    5.44,
    8.25,
    1.89,
    0.72,
    1.76,
    3.03,
    6.05,
    9.91
]
df_skills_base = pd.DataFrame({
    'clf': clf_names*6,
    'data': df_names,
    'variable': variable*18,
    'avg': avg_skills_base,
    'std': std_skills_base
    },
    columns=['clf', 'data', 'variable', 'avg', 'std']
)


# Skills Base
a4_dims = (9, 5)
fig, ax = plt.subplots(figsize=a4_dims)
g = sns.barplot(x='data', y='avg', hue='clf', data=df_skills_base, palette='Blues')
g.set(xlabel='Dataset', ylabel='Mean Precision in %')
ax.yaxis.set_ticks(np.arange(0, 100, 10))
ax.set(ylim=(0, 100))
ax.legend(loc='upper left', #bbox_to_anchor=(0.5, 1.05),
          fancybox=True, shadow=True, ncol=2)
plt.setp(ax.get_legend().get_texts(), fontsize='8')
# for p, avg in zip(ax.patches, df_skills_base['avg']):
#         ax.annotate("%.2f %% avg" % avg, (p.get_x() + p.get_width() / 1.5, p.get_height()),
#              ha='center', va='center', rotation=90, fontsize=6,
#              xytext=(0, 20), textcoords='offset points')
fig.tight_layout()
fig.savefig("skill_base.svg")


# LSTM

# item_lstm
df_names_skills_lstm = ["Blank", "Pre-Game", "In-Game", "Performance", "Patch Context"]
value = [28.4, 34.1, 36.8, 44.0, 40.6]
df_skills_lstm = pd.DataFrame({
    'data': df_names_skills_lstm,
    'accuracy': value
    },
    columns=['data', 'accuracy']
)
a4_dims = (9, 5)
fig, ax = plt.subplots(figsize=a4_dims)
g = sns.barplot(x='data', y='accuracy', data=df_skills_lstm, palette='Blues')
g.set(xlabel='Dataset', ylabel='Accuracy in %')
ax.set(ylim=(0, 100))
fig.tight_layout()
fig.savefig("items_lstm.svg")



# skills_lstm
df_names_skills_lstm = ["Blank", "Pre-Game", "Inventory", "Performance", "Patch Context"]
value = [61.1, 80.0, 81.1, 81.1, 75.0]
df_skills_lstm = pd.DataFrame({
    'data': df_names_skills_lstm,
    'accuracy': value
    },
    columns=['data', 'accuracy']
)
a4_dims = (9, 5)
fig, ax = plt.subplots(figsize=a4_dims)
g = sns.barplot(x='data', y='accuracy', data=df_skills_lstm, palette='Blues')
g.set(xlabel='Dataset', ylabel='Accuracy in %')
ax.set(ylim=(0, 100))
fig.tight_layout()
fig.savefig("skill_lstm.svg")


