import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("books_data_cleaned.csv")

print("=" * 70)
print("TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

print("\n1. MEANINGFUL QUESTIONS")
print("-" * 70)

questions = [
    "1. How many books and variables are present in the dataset?",
    "2. How are book ratings distributed?",
    "3. How are book prices distributed?",
    "4. Does the average book price vary with rating?",
    "5. Is there a relationship between book price and rating?",
    "6. Which categories contain the most books?",
    "7. Which categories have the highest average prices?",
    "8. Which categories have the highest average ratings?",
    "9. Are there missing values or duplicate records?",
    "10. Are there potential price outliers or anomalies?"
]

for question in questions:
    print(question)

print("\n\n2. DATA STRUCTURE")
print("-" * 70)

print("\nDataset Shape:")
print(df.shape)

print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Records:")
print(df.head())

print("\nBasic Statistical Summary:")
print(df.describe())

print("\n\n3. DATA QUALITY CHECK")
print("-" * 70)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:", df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nUnique Values in Each Column:")
print(df.nunique())

print("\n\n4. RATING ANALYSIS")
print("-" * 70)

rating_counts = df["Rating"].value_counts().sort_index()

print("\nRating Distribution:")
print(rating_counts)

print("\nAverage Rating:")
print(round(df["Rating"].mean(), 2))

print("\nMedian Rating:")
print(df["Rating"].median())

print("\n\n5. PRICE ANALYSIS")
print("-" * 70)

print("\nPrice Statistics:")
print(df["Price"].describe())

print("\nMinimum Price:", round(df["Price"].min(), 2))
print("Maximum Price:", round(df["Price"].max(), 2))
print("Average Price:", round(df["Price"].mean(), 2))
print("Median Price:", round(df["Price"].median(), 2))

print("\n\n6. AVERAGE PRICE BY RATING")
print("-" * 70)

avg_price_rating = (
    df.groupby("Rating")["Price"]
    .mean()
    .sort_index()
)

print("\nAverage Price for Each Rating:")
print(avg_price_rating.round(2))

print("\n\n7. HYPOTHESIS TEST - PRICE VS RATING")
print("-" * 70)

print("\nHypothesis:")
print("Book price may be related to book rating.")

correlation = df["Price"].corr(df["Rating"])

print("\nPearson Correlation:", round(correlation, 3))

if correlation > 0.3:
    print("Result: The data shows a positive relationship between price and rating.")
elif correlation < -0.3:
    print("Result: The data shows a negative relationship between price and rating.")
else:
    print("Result: The data shows a weak relationship between price and rating.")

print("\n\n8. BOOKS BY CATEGORY")
print("-" * 70)

category_counts = df["Category"].value_counts()

print("\nTotal Number of Categories:", df["Category"].nunique())

print("\nTop 10 Categories by Number of Books:")
print(category_counts.head(10))

print("\n\n9. AVERAGE PRICE BY CATEGORY")
print("-" * 70)

avg_price_category = (
    df.groupby("Category")["Price"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 Categories by Average Price:")
print(avg_price_category.head(10).round(2))

print("\n\n10. AVERAGE RATING BY CATEGORY")
print("-" * 70)

avg_rating_category = (
    df.groupby("Category")["Rating"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 Categories by Average Rating:")
print(avg_rating_category.head(10).round(2))

print("\n\n11. PRICE OUTLIER DETECTION")
print("-" * 70)

Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)

outliers = df[
    (df["Price"] < lower_bound) |
    (df["Price"] > upper_bound)
]

print("\nQ1:", round(Q1, 2))
print("Q3:", round(Q3, 2))
print("IQR:", round(IQR, 2))
print("Lower Bound:", round(lower_bound, 2))
print("Upper Bound:", round(upper_bound, 2))

print("\nNumber of Price Outliers:", len(outliers))

if len(outliers) > 0:
    print("\nPotential Price Outliers:")
    print(
        outliers[
            ["Title", "Category", "Price", "Rating"]
        ].head(10)
    )
else:
    print("\nNo potential price outliers were detected.")

print("\n\n12. CHART 1 - RATING DISTRIBUTION")

plt.figure(figsize=(8, 5))

plt.bar(
    rating_counts.index,
    rating_counts.values
)

plt.title(
    "Rating Distribution",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Rating", fontsize=11)
plt.ylabel("Number of Books", fontsize=11)

plt.xticks(rating_counts.index)

plt.grid(
    axis="y",
    alpha=0.2
)

plt.tight_layout()
plt.show()

print("\n\n13. CHART 2 - PRICE DISTRIBUTION")

plt.figure(figsize=(9, 5))

plt.hist(
    df["Price"],
    bins=30,
    edgecolor="black",
    alpha=0.75
)

median_price = df["Price"].median()

plt.axvline(
    median_price,
    linestyle="--",
    linewidth=2,
    label=f"Median Price: {median_price:.2f}"
)

plt.title(
    "Distribution of Book Prices",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Price", fontsize=11)
plt.ylabel("Number of Books", fontsize=11)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.2
)

plt.tight_layout()
plt.show()

print("\n\n14. CHART 3 - TOP 10 CATEGORIES")

top_categories = category_counts.head(10).sort_values()

plt.figure(figsize=(10, 6))

plt.barh(
    top_categories.index,
    top_categories.values
)

plt.title(
    "Top 10 Categories by Number of Books",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Number of Books", fontsize=11)
plt.ylabel("Category", fontsize=11)

plt.grid(
    axis="x",
    alpha=0.2
)

plt.tight_layout()
plt.show()

print("\n\n15. CHART 4 - AVERAGE PRICE BY RATING")

plt.figure(figsize=(8, 5))

plt.bar(
    avg_price_rating.index,
    avg_price_rating.values
)

plt.title(
    "Average Price by Rating",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Rating", fontsize=11)
plt.ylabel("Average Price", fontsize=11)

plt.xticks(avg_price_rating.index)

plt.grid(
    axis="y",
    alpha=0.2
)

plt.tight_layout()
plt.show()

print("\n\n16. CHART 5 - AVERAGE PRICE BY CATEGORY")

top_price_categories = (
    avg_price_category
    .head(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_price_categories.index,
    top_price_categories.values
)

plt.title(
    "Top 10 Categories by Average Price",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Average Price", fontsize=11)
plt.ylabel("Category", fontsize=11)

plt.grid(
    axis="x",
    alpha=0.2
)

plt.tight_layout()
plt.show()

print("\n\n17. CHART 6 - AVERAGE RATING BY CATEGORY")

top_rating_categories = (
    avg_rating_category
    .head(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_rating_categories.index,
    top_rating_categories.values
)

plt.title(
    "Top 10 Categories by Average Rating",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Average Rating", fontsize=11)
plt.ylabel("Category", fontsize=11)

plt.xlim(0, 5)

plt.grid(
    axis="x",
    alpha=0.2
)

plt.tight_layout()
plt.show()

print("\n\n18. CHART 7 - PRICE VS RATING")

np.random.seed(42)

jitter = np.random.uniform(
    -0.08,
    0.08,
    size=len(df)
)

rating_jittered = df["Rating"] + jitter

plt.figure(figsize=(9, 5))

plt.scatter(
    rating_jittered,
    df["Price"],
    alpha=0.45,
    s=30
)

z = np.polyfit(
    df["Rating"],
    df["Price"],
    1
)

p = np.poly1d(z)

x_line = np.linspace(
    df["Rating"].min(),
    df["Rating"].max(),
    100
)

plt.plot(
    x_line,
    p(x_line),
    linestyle="--",
    linewidth=2,
    label=f"Trend Line | Correlation: {correlation:.3f}"
)

plt.title(
    "Book Price vs Rating",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Rating", fontsize=11)
plt.ylabel("Price", fontsize=11)

plt.xticks(
    [1, 2, 3, 4, 5],
    ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
)

plt.legend()

plt.grid(
    alpha=0.2
)

plt.tight_layout()
plt.show()

print("\n\n" + "=" * 70)
print("TASK 2 EDA COMPLETED")
print("=" * 70)

print("\nDataset Records:", len(df))
print("Dataset Variables:", len(df.columns))
print("Total Categories:", df["Category"].nunique())
print("Average Price:", round(df["Price"].mean(), 2))
print("Average Rating:", round(df["Rating"].mean(), 2))
print("Price-Rating Correlation:", round(correlation, 3))
print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())
print("Potential Price Outliers:", len(outliers))

print("\nCharts Created: 7")

print("1. Rating Distribution")
print("2. Price Distribution")
print("3. Top 10 Categories by Number of Books")
print("4. Average Price by Rating")
print("5. Top 10 Categories by Average Price")
print("6. Top 10 Categories by Average Rating")
print("7. Price vs Rating")

print("\n" + "=" * 70)