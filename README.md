# Ringgit and Oil Prices

A small project I built to learn more about the Malaysian ringgit, and how it moves alongside oil prices and the US dollar.

**Live dashboard:** https://ringgit-oil-analysis.streamlit.app
*(Free hosting: if the app is asleep, click the button and it will be ready in about 30 seconds.)*

<img width="1439" alt="Dashboard overview" src="https://github.com/user-attachments/assets/f9752ab0-a8c3-48e2-9665-54f736aeb8d6" />
<img width="1440" alt="Dashboard section" src="https://github.com/user-attachments/assets/14fe7b0f-f942-4130-97cc-64a64997a74d" />
<img width="1440" alt="Dashboard section" src="https://github.com/user-attachments/assets/21ecf3a6-3095-46f8-ad0d-4e6accba7cc2" />
<img width="1440" alt="Dashboard section" src="https://github.com/user-attachments/assets/93ea245f-c655-4544-bf59-d3ec1592d3ee" />
<img width="1440" alt="Dashboard section" src="https://github.com/user-attachments/assets/12316796-fb13-4f6f-8960-3ef7cc416bd1" />

## Why I made this

Malaysia sells oil and gas, and buyers pay in US dollars. I often heard that this links the oil price to the ringgit, and I wanted to see what the data actually shows. I also wanted to practise working with real financial data in Python, from loading and joining datasets to building and deploying a dashboard.

## What it shows

- How the ringgit is doing against the US dollar, Chinese yuan and Singapore dollar, compared with 1 month, 1 year or 5 years ago
- The price of these currencies and oil on any day since July 2005
- How the ringgit and oil prices have moved together over time
- How the ringgit has changed against Malaysia's top trading partners and ASEAN neighbours

The dashboard updates automatically with the latest published rates.

## Data

| Data | Source |
|---|---|
| Daily ringgit exchange rates (12pm middle rate) | Bank Negara Malaysia, via [data.gov.my](https://data.gov.my/data-catalogue/exchangerates_daily_1200) |
| Brent crude oil price (USD per barrel) | [FRED](https://fred.stlouisfed.org/series/DCOILBRENTEU) |
| US dollar index | [FRED](https://fred.stlouisfed.org/series/DTWEXBGS) |

The data starts on 22 July 2005, when the ringgit stopped being fixed at RM3.80 per US dollar.

## What's in this repo

| File | What it is |
|---|---|
| `analysis.ipynb` | The analysis, step by step: loading and cleaning the data, charts, correlation, and a simple regression model |
| `app.py` | The Streamlit dashboard |
| `requirements.txt` | Python packages needed to run the dashboard |
| `*.csv`, `*.parquet` | Saved copies of the data, used as a backup |

## What I did

- Cleaned and joined three daily datasets by date
- Explored trends with charts and correlation, and compared different time periods
- Built a simple linear regression model, trained on 2005 to 2021 and tested on 2022 onwards
- Checked whether the model could predict next month's rate better than a simple "no change" guess (it couldn't, which is common for exchange rates)
- Built and deployed a dashboard with Streamlit

## Limitations

- Moving together does not mean one causes the other
- Oil and the US dollar affect each other, so their effects can't be fully separated
- Many other things move the ringgit (interest rates, trade, politics) that are not included here

## Tools

Python, pandas, matplotlib, scikit-learn, Plotly, Streamlit

## Run it locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

Made by [aamrhh](https://github.com/aamrhh)
