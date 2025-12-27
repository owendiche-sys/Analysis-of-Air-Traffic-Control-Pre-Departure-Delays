"""
Name: Owen Nda Diche
Student ID: 24152506


AI assistance used: ChatGPT (OpenAI, 2025) 
for code clarification and proofreading
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# Plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

# Creating results folder
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Loading Dataset
df = pd.read_csv("combined_pre_departure_delay_dataset_2017_2023.csv")

print(df.head())
print(df.tail())
print(df.describe())


# Data Cleaning & Prepartion 
df.columns = df.columns.str.lower()

DATE_COL = "flt_date"
DELAY_COLS = ["dly_atc_pre_2", "dly_atc_pre_3"]

# Convert date column
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL])

# Convert delay columns to numeric
for col in DELAY_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Create total delay
df["total_delay"] = df[DELAY_COLS].sum(axis=1)

# Time features
df["month"]      = df[DATE_COL].dt.month
df["month_name"] = df[DATE_COL].dt.month_name()

print("Dataset Shape:", df.shape)

# Summary Statistics Table
summary_table = (
    df.groupby("year")["total_delay"]
      .agg(["mean", "median", "min", "max", "std", "count"])
      .round(2)
)

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')

table = ax.table(
    cellText=summary_table.values,
    colLabels=summary_table.columns,
    rowLabels=summary_table.index,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2) 

output_path = RESULTS_DIR / "summary table.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Summary Statistics table saved as:", output_path)
print("\nSummary Statistics by Year:")
print(summary_table)

# Figure 1: Daily Delays (7-day Rolling Mean)
plt.figure(figsize=(14, 7))

for yr in sorted(df["year"].unique()):
    sub = df[df["year"] == yr].sort_values(DATE_COL)
    rolling = sub.set_index(DATE_COL)["total_delay"].rolling(
        7, min_periods=1).mean()
    plt.plot(rolling.index, rolling.values, label=str(yr))

plt.title(
    "Daily Pre-Departure Delay (7-day Rolling Mean)",
    fontweight='bold',
    fontsize= 20
)
plt.xlabel("Year", fontsize= 18,fontweight='bold' )
plt.ylabel("Total Delay (minutes)", fontsize= 16, fontweight='bold')
plt.xticks( fontsize= 18, fontweight='bold')
plt.yticks( fontsize= 18, fontweight='bold')
plt.legend()
plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

plt.tight_layout()
plt.savefig(RESULTS_DIR / "figure 1.png", dpi=300)
plt.show()

#Figure 2: Monthly Average Delay Per Year
monthly = (
    df.groupby(["year", "month"])["total_delay"]
      .mean()
      .reset_index()
)

plt.figure(figsize=(14, 7))
sns.lineplot(
    data=monthly, 
    x="month", 
    y="total_delay", 
    hue="year",
    marker="o",
    palette="tab20",
    linewidth= 2.5
)

plt.title(
    "Monthly Average Pre-Departure Delay",
    fontweight='bold',
    fontsize= 20
)
plt.xlabel("Month", fontsize= 18, fontweight='bold')
plt.ylabel("Average Delay (minutes)", fontsize= 18, fontweight='bold')
plt.xticks(
    range(1, 13),
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    fontsize= 18, 
    fontweight='bold'
)
plt.yticks( fontsize= 18, fontweight='bold')
plt.legend(
    title="Year",
    ncol=1,                          
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0
)
plt.tight_layout()

plt.savefig(RESULTS_DIR / "figure 2.png", dpi=300)
plt.show()

# Figure 3: Average Monthly Delay by Year (Grouped Bar Chart)
delay_per_year = (
    df.groupby("year")["total_delay"]
      .sum()
      .reset_index()
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=delay_per_year,
    x="year",
    y="total_delay",
    hue="year",
    palette="tab10",
    dodge=False,
    legend=False
)

plt.title(
    "Total ATC Pre-Departure Delay per Year (2017–2023)",
    fontsize= 20,
    fontweight='bold'
    )
plt.xlabel("Year", fontsize= 18, fontweight='bold')
plt.ylabel("Total Delay (minutes)",fontsize= 18, fontweight='bold')
plt.yticks( fontsize= 18, fontweight='bold')
plt.xticks( fontsize= 18, fontweight='bold')

plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
plt.savefig(RESULTS_DIR / "figure 3.png", dpi=300)
plt.show()


# Figure 4: Top 10 Airports by Total Delay
# Creating label
df["airport_label"] = df["apt_name"] + " (" + df["apt_icao"] + ")"

# Computing totals
airport_delay = (
    df.groupby("airport_label")["total_delay"]
      .sum()
      .sort_values(ascending=False)
)

top10 = airport_delay.head(10).reset_index()
top10.columns = ["airport_label", "total_delay"]

plt.figure(figsize=(15, 9))
sns.barplot(
    data=top10,
    x="total_delay",
    y="airport_label",
    hue="airport_label",
    palette="tab10",
    dodge=False,
    legend=False
)

plt.title(
    "Top 10 Airports by Total ATC Pre-Departure Delay (2017-2023)",
    fontsize= 22,
    fontweight='bold'
)
plt.xlabel("Total Delay (minutes)", fontsize= 20, fontweight='bold')
plt.ylabel("Airport", fontsize= 20, fontweight='bold')
plt.xticks( fontsize= 20, fontweight='bold', rotation=45)
plt.yticks( fontsize= 20, fontweight='bold')
plt.gca().xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "figure 4.png", 
    dpi=300
)
plt.show()
