# Assignment 2 - Statistics and Trends
# Name: Ruzen Maharjan
# Student ID: 24130873
# Date: March 2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#-----------------------------
# function to load and clean the data
# -----------------------------
def load_worldbank_data(filename, skiprows=0):

    df = pd.read_csv(filename, skiprows=skiprows, encoding="utf-8-sig")

    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    df = df.drop(columns=["Country Code", "Indicator Name", "Indicator Code"], errors='ignore')
    df = df.set_index("Country Name")

    df = df.apply(pd.to_numeric, errors="coerce")

    df_t = df.T

    return df, df_t


# -----------------------------
# Loading population, agricultural land and co2 datasets
# -----------------------------
pop_df, pop_t = load_worldbank_data("API_SP.POP.TOTL_DS2_en_csv_v2_84031.csv")

agri_df, agri_t = load_worldbank_data("API_AG.LND.AGRI.K2_DS2_en_csv_v2_46880.csv")

co2_df, co2_t = load_worldbank_data(
    "API_EN.GHG.CO2.FE.MT.CE.AR5_DS2_en_csv_v2_12590.csv",
    skiprows=4
)

print("data loaded successfully")


# -----------------------------
# SELECT COUNTRIES
# -----------------------------
countries = ["United States", "Germany", "China",
             "Brazil", "Ethiopia", "Bangladesh"]


# -----------------------------
# AGRICULTURAL LAND PER CAPITA
# -----------------------------
agri_per_cap = agri_df.loc[countries] / pop_df.loc[countries] * 1000000


# -----------------------------
# CORRELATION
# -----------------------------
print("\nCorrelation: CO2 vs Agricultural Land")

for country in countries:
    co2_row = co2_df.loc[country].dropna()
    agri_row = agri_per_cap.loc[country].dropna()

    common = co2_row.index.intersection(agri_row.index)
    co2_row = co2_row[common]
    agri_row = agri_row[common]

    if len(co2_row) > 3:
        print(country, ":", co2_row.corr(agri_row))


# -----------------------------
# RATE OF CHANGE
# -----------------------------
print("\nRate of Change (CO2)")

for country in countries:
    row = co2_df.loc[country].dropna()
    row.index = row.index.astype(int)

    early = row[(row.index >= 2000) & (row.index <= 2010)]
    late = row[(row.index >= 2010) & (row.index <= 2020)]

    if len(early) > 1:
        early_rate = (early.iloc[-1] - early.iloc[0]) / 10
    else:
        early_rate = 0

    if len(late) > 1:
        late_rate = (late.iloc[-1] - late.iloc[0]) / 10
    else:
        late_rate = 0

    print(country, ": early =", round(early_rate, 2),
          " late =", round(late_rate, 2))


# -----------------------------
# BOOTSTRAPPING
# -----------------------------
china_co2 = co2_df.loc["China"].dropna()
china_agri = agri_per_cap.loc["China"].dropna()

common = china_co2.index.intersection(china_agri.index)
china_co2 = china_co2[common].values
china_agri = china_agri[common].values

means = []

for i in range(300):
    sample = np.random.choice(china_co2, size=len(china_co2), replace=True)
    means.append(np.mean(sample))

print("\nBootstrapping (China CO2 mean approx):", round(np.mean(means), 2))


# -----------------------------
# GRAPH 1 - CO2 over time
# -----------------------------
co2_t[countries].plot()
plt.title("CO2 Emissions Over Time")
plt.xlabel("Year")
plt.ylabel("CO2 Emissions")
plt.savefig("figure1_co2_trend.png", dpi=150)
plt.show()


# -----------------------------
# GRAPH 2 - Agri vs CO2
# ------------------------------
plt.figure()

for country in countries:
    co2_row = co2_df.loc[country].dropna()
    agri_row = agri_per_cap.loc[country].dropna()

    # match same years
    common = co2_row.index.intersection(agri_row.index)
    co2_row = co2_row[common]
    agri_row = agri_row[common]

    if len(co2_row) > 2:
        plt.scatter(agri_row, co2_row, label=country)

plt.xlabel("Agricultural Land per capita")
plt.ylabel("CO2 Emissions")
plt.title("Agricultural Land vs CO2 Emissions")
plt.legend()
plt.savefig("figure2_agriculture.png", dpi=150)
plt.show()

# -----------------------------
# GRAPH 3 - Rate comparison
# -----------------------------
early_rates = []
late_rates = []

for country in countries:
    row = co2_df.loc[country].dropna()
    row.index = row.index.astype(int)

    early = row[(row.index >= 2000) & (row.index <= 2010)]
    late = row[(row.index >= 2010) & (row.index <= 2020)]

    early_rates.append((early.iloc[-1] - early.iloc[0]) / 10 if len(early) > 1 else 0)
    late_rates.append((late.iloc[-1] - late.iloc[0]) / 10 if len(late) > 1 else 0)

x = np.arange(len(countries))

plt.bar(x - 0.2, early_rates, width=0.4, label="Early")
plt.bar(x + 0.2, late_rates, width=0.4, label="Recent")

plt.xticks(x, countries, rotation=20)
plt.title("CO2 Change Over Time")
plt.ylabel("Rate of Change")
plt.legend()
plt.savefig("figure3_rate.png", dpi=150)
plt.show()


# -----------------------------
# GRAPH 4 - Population vs CO2
# -----------------------------
plt.figure()

for country in countries:
    co2_row = co2_df.loc[country].dropna()
    pop_row = pop_df.loc[country].dropna()

    common = co2_row.index.intersection(pop_row.index)

    plt.scatter(pop_row[common], co2_row[common], label=country)

plt.xlabel("Population")
plt.ylabel("CO2 Emissions")
plt.title("Population vs CO2 Emissions")
plt.legend()
plt.savefig("figure4_population.png", dpi=150)
plt.show()


