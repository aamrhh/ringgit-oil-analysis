"""
How Oil Prices Move the Ringgit: a two-page dashboard.

Run it from the project folder (with the venv on):
    streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------
# Settings: colours, dates, data links
# ---------------------------------------------------------------
st.set_page_config(page_title="Ringgit vs Oil", page_icon=":material/currency_exchange:", layout="wide")

TEXT = "#0B2545"      # titles and text
RINGGIT = "#1E3A5F"   # ringgit is always navy
OIL = "#059669"       # oil is always emerald
DOLLAR = "#B08D57"    # US dollar is always gold
WARN = "#B91C1C"      # red only for warnings / the break
MUTED = "#64748B"     # small grey text
GRID = "#E2E8F0"      # light grid lines

PEG_END = "2005-07-22"   # ringgit stopped being fixed at RM3.80
TEST_START = "2022-01-01"   # train on everything before 2022, test on 2022 to now
BREAK = "2022-01-01"

FX_URL = "https://storage.data.gov.my/finsector/exr/daily_1200.parquet"
OIL_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"
DOLLAR_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS"

NEIGHBOURS = {
    "CNY": "Chinese yuan",
    "SGD": "Singapore dollar",
    "USD": "US dollar",
    "THB": "Thai baht",
    "IDR": "Indonesian rupiah",
    "VND": "Vietnamese dong",
    "PHP": "Philippine peso",
    "BND": "Brunei dollar",
    "KHR": "Cambodian riel",
    "MMK": "Myanmar kyat",
}
TOP_PARTNERS = ["USD", "CNY", "SGD"]  # Malaysia's 3 biggest trading partners in 2025 (DOSM)

# A little styling for the number cards
st.markdown(
    f"""
    <style>
    [data-testid="stMetric"] {{
        background: #FFFFFF; border: 1px solid {GRID};
        border-radius: 12px; padding: 16px 20px;
    }}
    [data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.9rem; }}
    h1, h2, h3 {{ color: {TEXT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# 1. Load the data (from my saved files, or the internet if missing)
# ---------------------------------------------------------------
def read_fred(local_file, url, name):
    """Read a FRED series (newest online copy, or my saved file if offline)."""
    try:
        data = pd.read_csv(url)
        data.columns = ["date", name]
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date")
    except Exception:
        data = pd.read_csv(local_file, index_col=0, parse_dates=True)
        data.columns = [name]
    data[name] = pd.to_numeric(data[name], errors="coerce")
    return data.sort_index()


@st.cache_data(ttl=6 * 60 * 60)  # download fresh data at most every 6 hours
def load_data():
    try:
        fx = pd.read_parquet(FX_URL)
    except Exception:
        fx = pd.read_parquet("fx_raw.parquet")
    fx["date"] = pd.to_datetime(fx["date"])
    fx = fx[fx["rate_type"] == "middle"].set_index("date").sort_index()
    fx = fx[fx.index >= PEG_END]

    oil = read_fred("oil_raw.csv", OIL_URL, "Brent_USD")
    dollar = read_fred("dollar_raw.csv", DOLLAR_URL, "USD_index")

    usd = fx[["USD"]].rename(columns={"USD": "MYR_per_USD"}).dropna()
    df = usd.join(oil, how="inner").dropna()
    df2 = df.join(dollar, how="inner").dropna()
    return fx, oil.dropna(), df, df2, dollar.dropna()


# ---------------------------------------------------------------
# 2. All the analysis (same steps as the notebook)
# ---------------------------------------------------------------
def corr(data):
    return data["MYR_per_USD"].corr(data["Brent_USD"])


def split(data):
    return data[data.index < BREAK], data[data.index >= BREAK]


@st.cache_data
def analyse(df, df2, fx):
    r = {}

    # Correlation of prices
    before, after = split(df)
    r["corr"] = {"All years": corr(df), "Before 2022": corr(before), "2022 to now": corr(after)}

    # Correlation of monthly changes
    changes = df.resample("ME").last().pct_change().dropna()
    before_ch, after_ch = split(changes)
    r["corr_changes"] = {
        "All years": corr(changes),
        "Before 2022": corr(before_ch),
        "2022 to now": corr(after_ch),
    }

    # Lag test: does the ringgit react later? (before 2022)
    r["lag"] = {
        lag: before_ch["MYR_per_USD"].corr(before_ch["Brent_USD"].shift(lag))
        for lag in [0, 1, 2, 3]
    }

    # Model 1: oil only
    train1 = df[df.index < TEST_START]
    model1 = LinearRegression().fit(train1[["Brent_USD"]], train1["MYR_per_USD"])

    # Model 2: oil + US dollar
    inputs = ["Brent_USD", "USD_index"]
    train2 = df2[df2.index < TEST_START]
    model2 = LinearRegression().fit(train2[inputs], train2["MYR_per_USD"])

    periods = {
        "Train (studied)": lambda d: d[d.index < TEST_START],
        "Test: 2022–now": lambda d: d[d.index >= TEST_START],
    }
    r["mae"] = {}
    for name, pick in periods.items():
        p1, p2 = pick(df), pick(df2)
        r["mae"][name] = {
            "Oil only": mean_absolute_error(p1["MYR_per_USD"], model1.predict(p1[["Brent_USD"]])),
            "Oil + US dollar": mean_absolute_error(p2["MYR_per_USD"], model2.predict(p2[inputs])),
        }

    guesses = df2[["MYR_per_USD"]].copy()
    guesses["Oil only"] = model1.predict(df2[["Brent_USD"]])
    guesses["Oil + US dollar"] = model2.predict(df2[inputs])
    r["guesses"] = guesses

    r["model1"] = (model1.intercept_, model1.coef_[0])
    r["model2"] = (model2.intercept_, model2.coef_[0], model2.coef_[1])
    r["model2_obj"] = model2

    # Fair comparison: effect of a typical move
    r["typical_oil"] = model2.coef_[0] * df2["Brent_USD"].std()
    r["typical_dollar"] = model2.coef_[1] * df2["USD_index"].std()
    r["oil_range"] = (df2["Brent_USD"].min(), df2["Brent_USD"].max())
    r["dollar_range"] = (df2["USD_index"].min(), df2["USD_index"].max())

    # Prediction test: next month vs the lazy guess
    m = df2[["MYR_per_USD", "Brent_USD", "USD_index"]].resample("ME").last()
    m["next_ringgit"] = m["MYR_per_USD"].shift(-1)
    m = m.dropna()
    feats = ["MYR_per_USD", "Brent_USD", "USD_index"]
    tr, te = m[m.index < TEST_START], m[m.index >= TEST_START]
    fmodel = LinearRegression().fit(tr[feats], tr["next_ringgit"])
    r["forecast"] = {
        "months": len(te),
        "lazy": mean_absolute_error(te["next_ringgit"], te["MYR_per_USD"]),
        "model": mean_absolute_error(te["next_ringgit"], fmodel.predict(te[feats])),
    }

    return r


def join_names(names):
    """['A', 'B', 'C'] -> 'A, B and C'."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def value_on(series, day):
    """(date, value) on the given day, or the closest working day before it."""
    s = series.dropna()
    before = s[s.index <= day]
    if before.empty:
        return s.index[0], s.iloc[0]
    return before.index[-1], before.iloc[-1]


def neighbours_change(fx, start):
    """% change of the ringgit vs each currency, from `start` to its latest day (+ = ringgit weaker)."""
    out = {}
    for cur in NEIGHBOURS:
        if cur not in fx.columns or fx[cur].dropna().empty:
            continue
        _, then = value_on(fx[cur], start)
        out[cur] = (fx[cur].dropna().iloc[-1] / then - 1) * 100
    first_day, _ = value_on(fx["USD"], start)
    return pd.Series(out).sort_values(), first_day


# ---------------------------------------------------------------
# 3. Chart helpers
# ---------------------------------------------------------------
def style(fig, height=420):
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=TEXT, size=13),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.08, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def mark_date(fig, date, text, colour=MUTED, y=1.0):
    """Dotted vertical line with a short label (y = label height, so labels don't overlap)."""
    fig.add_shape(type="line", x0=date, x1=date, y0=0, y1=1, yref="paper",
                  line=dict(color=colour, width=1, dash="dot"))
    fig.add_annotation(x=date, y=y, yref="paper", yanchor="top", text=text, xanchor="left",
                       showarrow=False, font=dict(size=10, color=colour),
                       bgcolor="rgba(255,255,255,0.85)")


def rm(value):
    """Money with a sign in front: +RM0.47 or −RM0.06."""
    return f"{'+' if value >= 0 else '−'}RM{abs(value):.2f}"


def pct_word(value):
    if abs(value) < 0.05:
        return "about the same"
    return f"{abs(value):.1f}% {'weaker' if value > 0 else 'stronger'}"


# ---------------------------------------------------------------
# 4. The dashboard
# ---------------------------------------------------------------
fx, oil, df, df2, dollar = load_data()
r = analyse(df, df2, fx)

latest_ringgit_date = fx["USD"].dropna().index.max()
latest_ringgit = fx["USD"].dropna().iloc[-1]
year_ago_ringgit = fx["USD"].dropna().asof(latest_ringgit_date - pd.DateOffset(years=1))
latest_oil_date = oil.index.max()
latest_oil = oil["Brent_USD"].iloc[-1]
year_ago_oil = oil["Brent_USD"].asof(latest_oil_date - pd.DateOffset(years=1))

st.title("How oil prices move the ringgit")
st.caption(
    f"Latest data: ringgit {latest_ringgit_date:%d %b %Y} · oil {latest_oil_date:%d %b %Y} · "
    f"US dollar index {dollar.index.max():%d %b %Y}. Updates automatically; rates are published on working days only. "
    "Sources: Bank Negara Malaysia (data.gov.my) and FRED."
)

page1, page2 = st.tabs(["Understand the market", "How I did the analysis"])

# ===============================================================
# PAGE 1: for everyone
# ===============================================================
with page1:

    CARD_H = 260
    latest_day = fx["USD"].dropna().index.max()
    PERIODS = {
        "1 month ago": pd.DateOffset(months=1),
        "1 year ago": pd.DateOffset(years=1),
        "5 years ago": pd.DateOffset(years=5),
    }
    st.markdown("#### Is the ringgit stronger or weaker than before?")
    period = st.radio("Compare today with", list(PERIODS), index=0, horizontal=True)
    start_day = latest_day - PERIODS[period]

    n, n_start = neighbours_change(fx, start_day)
    compare_word = period
    then_word = period

    def window(series, begin):
        """The part of a series inside the chosen period, thinned out for a small trend line."""
        part = series.dropna()
        part = part[part.index >= begin]
        if len(part) > 140:
            part = part.resample("W").last().dropna()
        return part.round(3).tolist()

    def card(col, label, value, delta, colour, arrow, note, chart=None, height=CARD_H, tip=None):
        """A number card with a short change line and a small note that can wrap."""
        with col.container(border=True, height=height):
            st.metric(label, value, delta, delta_color=colour, delta_arrow=arrow,
                      chart_data=chart, help=tip)
            st.caption(note)

    # ---- Number cards ----------------------------------------------------
    usd_series = fx["USD"].dropna()
    _, ringgit_then = value_on(usd_series, n_start)
    oil_series = oil["Brent_USD"]
    oil_day, oil_then = value_on(oil_series, n_start)
    fx_diff = latest_ringgit - ringgit_then
    oil_diff = latest_oil - oil_then
    label_then = compare_word[0].upper() + compare_word[1:]

    st.caption("Malaysia's three biggest trading partners in 2025 were China, Singapore and the United "
               "States, together about 43% of all trade (DOSM).")
    tips = {
        "USD": "Oil and most world trade are priced in US dollars. Malaysia's 3rd biggest trading partner.",
        "CNY": "China has been Malaysia's biggest trading partner for 17 years in a row.",
        "SGD": "Malaysia's 2nd biggest trading partner and closest neighbour.",
    }
    for col, cur in zip(st.columns(3), TOP_PARTNERS):
        v = n[cur]
        card(col, f"Ringgit vs {NEIGHBOURS[cur]}",
             "About the same" if abs(v) < 0.05 else f"{abs(v):.1f}%",
             None if abs(v) < 0.05 else ("Ringgit stronger" if v < 0 else "Ringgit weaker"),
             "green" if v < 0 else "red", "up" if v < 0 else "down",
             f"Compared with {compare_word}", window(fx[cur], n_start), tip=tips[cur])

    # ---- Wallet ----------------------------------------------------------
    st.markdown("#### What this means for your wallet")
    rates = fx[TOP_PARTNERS].dropna()
    then_rates = rates[rates.index <= n_start].iloc[-1]
    now_rates = rates.iloc[-1]
    for col, cur in zip(st.columns(3), TOP_PARTNERS):
        code = cur
        now_amt, then_amt = 1000 / now_rates[cur], 1000 / then_rates[cur]
        diff = now_amt - then_amt
        same = abs(diff) < 0.5
        card(col, f"Today, RM1,000 buys you", f"{code} {now_amt:,.0f}",
             f"About the same as {compare_word}" if same
             else f"{code} {abs(diff):,.0f} {'more' if diff > 0 else 'less'} than {compare_word}",
             "gray" if same else ("green" if diff > 0 else "red"),
             "off" if same else ("up" if diff > 0 else "down"),
             f"{label_then}: RM1,000 bought {code} {then_amt:,.0f}", height=170)

    # ---- Look up any day -------------------------------------------------
    st.markdown("#### Look up any day")
    d1, d2 = st.columns([1, 3])
    today_my = pd.Timestamp.now(tz="Asia/Kuala_Lumpur").tz_localize(None).normalize()
    picked = d1.date_input(
        "Choose a date",
        value=today_my.date(),
        min_value=pd.Timestamp(PEG_END).date(),
        max_value=today_my.date(),
        format="DD/MM/YYYY",
    )
    pick_day = pd.Timestamp(picked)

    def day_and_before(series, day):
        """Value on the day (or closest working day before), the working day before it, and 30 days of history."""
        s_ = series.dropna()
        s_ = s_[s_.index <= day]
        if len(s_) < 2:
            s_ = series.dropna().iloc[:2]
        return s_.index[-1], s_.iloc[-1], s_.index[-2], s_.iloc[-2], s_.iloc[-30:].round(3).tolist()

    with d2:
        u_day, u_val, u_prev_day, u_prev, u_hist = day_and_before(fx["USD"], pick_day)
        st.markdown(
            f"<div style='display:inline-block;background:{RINGGIT};color:#FFFFFF;border-radius:8px;"
            f"padding:6px 14px;margin:24px 0 6px;font-weight:600;'>{pick_day:%A, %d %B %Y}</div>",
            unsafe_allow_html=True,
        )
        if u_day.normalize() == pick_day.normalize():
            st.caption("Rates on this day. Small line = the 30 working days up to this date.")
        else:
            if pick_day.dayofweek >= 5:
                why = "Bank Negara does not publish exchange rates on weekends."
            elif pick_day > latest_day:
                why = ("The rate for this day has not been published yet. "
                       "Bank Negara publishes it around 12pm on working days.")
            else:
                why = "Bank Negara did not publish a rate on this day (likely a public holiday)."
            st.caption(f"{why} Showing the latest rate before this date: **{u_day:%A, %d %B %Y}**.")

    l1, l2, l3, l4 = st.columns(4)
    for col, cur in zip([l1, l2, l3], TOP_PARTNERS):
        c_day, c_val, c_prev_day, c_prev, c_hist = day_and_before(fx[cur], pick_day)
        change = c_val - c_prev
        same = abs(change) < 0.00005
        card(col, f"1 {NEIGHBOURS[cur]}", f"RM{c_val:.4f}" if c_val < 1 else f"RM{c_val:.2f}",
             None if same else ("Ringgit stronger" if change < 0 else "Ringgit weaker"),
             "green" if change < 0 else "red", "up" if change < 0 else "down",
             f"Rate from **{c_day:%a, %d %b}**. Compared with {c_prev_day:%a, %d %b}: "
             + (f"RM{c_prev:.4f}" if c_prev < 1 else f"RM{c_prev:.2f}"),
             c_hist, height=230)
    o_day, o_val, o_prev_day, o_prev, o_hist = day_and_before(oil["Brent_USD"], pick_day)
    o_change = o_val - o_prev
    card(l4, "Oil, per barrel", f"USD {o_val:.0f}",
         f"USD {abs(o_change):.1f} {'higher' if o_change > 0 else 'lower'}",
         "gray", "up" if o_change > 0 else "down",
         f"Price from **{o_day:%a, %d %b}** (oil data comes a few days late). "
         f"Compared with {o_prev_day:%a, %d %b}: USD {o_prev:.0f}",
         o_hist, height=230)

    # ---- Main chart ------------------------------------------------------
    st.subheader("Oil and the ringgit over time")
    st.markdown(
        f"<span style='color:{RINGGIT}'>**Navy line:**</span> ringgit needed for 1 US dollar "
        "(line going up = ringgit getting weaker). "
        f"<span style='color:{OIL}'>**Green line:**</span> world oil price, in US dollars per barrel. "
        "For many years they moved like a mirror: when oil rose, the ringgit got stronger. "
        "Since 2022, that mirror has broken.",
        unsafe_allow_html=True,
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df["MYR_per_USD"], name="Ringgit per 1 US dollar",
                             line=dict(color=RINGGIT, width=2),
                             hovertemplate="RM%{y:.2f}"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["Brent_USD"], name="Oil (USD per barrel)",
                             line=dict(color=OIL, width=1.5),
                             hovertemplate="USD %{y:.0f}"), secondary_y=True)
    mark_date(fig, "2008-09-01", "2008 crash", y=0.99)
    mark_date(fig, "2014-12-01", "2014 crash", y=0.99)
    mark_date(fig, "2020-04-01", "COVID", y=0.88)
    mark_date(fig, "2022-06-01", "Strong USD", WARN, y=0.99)
    style(fig, 400)
    fig.update_xaxes(
        range=["2014-01-01", df.index.max()],
        rangeselector=dict(
            buttons=[
                dict(count=5, label="5 years", step="year", stepmode="backward"),
                dict(count=12, label="12 years", step="year", stepmode="backward"),
                dict(step="all", label="Since 2005"),
            ],
            bgcolor="#F1F5F9", activecolor="#CBD5E1", x=0, y=1.1, font=dict(size=11),
        ),
    )
    fig.update_layout(margin=dict(l=5, r=5, t=50, b=5))
    fig.update_yaxes(tickprefix="RM", secondary_y=False)
    fig.update_yaxes(tickprefix="$", showgrid=False, secondary_y=True)
    with st.container(border=True):
        st.plotly_chart(fig, width="stretch")

    # ---- How it works (flow) --------------------------------------------
    st.subheader("How oil can move the ringgit")
    steps = [
        ("Oil price goes up", "Malaysian oil and gas are priced from Brent, the world's main oil price."),
        ("More US dollars come in", "Buyers pay Petronas and other companies in US dollars."),
        ("Dollars are changed into ringgit", "Companies need ringgit to pay salaries, tax and local suppliers."),
        ("Ringgit gets stronger", "More people want ringgit, so you need fewer ringgit for 1 US dollar."),
    ]
    items = []
    for i, (title, text) in enumerate(steps, start=1):
        dot = OIL if i == 4 else RINGGIT
        items.append(
            f"<div style='flex:1;min-width:150px;position:relative;padding-top:4px;'>"
            f"<div style='width:34px;height:34px;border-radius:50%;background:{dot};color:#FFFFFF;"
            f"display:flex;align-items:center;justify-content:center;font-weight:600;"
            f"position:relative;z-index:1;'>{i}</div>"
            f"<div style='font-weight:600;color:{TEXT};margin:12px 0 4px;'>{title}</div>"
            f"<div style='font-size:0.9rem;color:{MUTED};line-height:1.45;padding-right:12px;'>{text}</div></div>"
        )
    st.markdown(
        f"<div style='position:relative;'>"
        f"<div style='position:absolute;top:21px;left:17px;right:17px;height:2px;background:{GRID};'></div>"
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;position:relative;'>{''.join(items)}</div></div>"
        f"<p style='color:{MUTED};margin-top:14px;'>When oil falls, the same chain runs backwards: "
        "fewer US dollars come in, and the ringgit usually gets weaker.</p>",
        unsafe_allow_html=True,
    )

    # ---- Neighbours ------------------------------------------------------
    st.subheader("Is it only the US dollar?")
    st.caption("The ringgit against Malaysia's top 3 trading partners and its ASEAN neighbours.")
    left, right = st.columns([3, 2])
    with left:
        colours = [RINGGIT if v > 0 else OIL for v in n.values]
        bar = go.Figure(go.Bar(
            x=n.values, y=[NEIGHBOURS[c] for c in n.index], orientation="h",
            marker_color=colours, text=[f"{v:+.1f}%" for v in n.values],
            textposition="outside", hovertemplate="%{x:.1f}%<extra></extra>",
        ))
        style(bar, 420)
        bar.update_layout(hovermode="closest", margin=dict(l=10, r=40, t=20, b=10))
        bar.update_xaxes(title_text=f"Change compared with {compare_word} (right = ringgit weaker, left = ringgit stronger)",
                         showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=MUTED)
        with st.container(border=True):
            st.plotly_chart(bar, width="stretch")
    with right:
        worse = [NEIGHBOURS[c] for c in n.index if c != "USD" and n[c] > max(n["USD"], 0)]
        stronger = [NEIGHBOURS[c] for c in n.index if n[c] < 0 and c != "USD"]
        points = [f"Compared with {compare_word}, the ringgit is **{pct_word(n['USD'])}** against the US dollar."]
        if worse:
            points.append(f"It did worse against the {join_names(worse)}. So it is not only about "
                          "the US dollar: some neighbours did better than the ringgit.")
        if stronger:
            points.append(f"It got **stronger** against the {join_names(stronger)}.")
        st.markdown("\n".join(f"- {p}" for p in points))

    # ---- Takeaways (follow the period chosen at the top) -----------------
    st.subheader(f"Key takeaways: compared with {compare_word}")
    TAKE_H = 430
    t1, t2, t3 = st.columns(3)

    def mini_lines(data, cols, names, colours, dual=False):
        f = make_subplots(specs=[[{"secondary_y": dual}]])
        for i, (col, name, colour) in enumerate(zip(cols, names, colours)):
            f.add_trace(go.Scatter(x=data.index, y=data[col], name=name,
                                   line=dict(color=colour, width=2),
                                   hovertemplate="%{y:.2f}"),
                        secondary_y=(dual and i == 1))
        style(f, 260)
        f.update_layout(margin=dict(l=5, r=5, t=10, b=5), font=dict(size=11), legend=dict(y=-0.15))
        if dual:
            f.update_yaxes(showgrid=False, secondary_y=True)
        return f

    recent = df[df.index >= n_start]
    ringgit_word = pct_word(n["USD"])

    with t1.container(border=True, height=TAKE_H):
        o_then, o_now = recent["Brent_USD"].iloc[0], recent["Brent_USD"].iloc[-1]
        o_pct = (o_now / o_then - 1) * 100
        st.markdown(f"**Oil went {'up' if o_pct > 0 else 'down'} {abs(o_pct):.0f}%**")
        st.caption(f"Oil went from USD {o_then:.0f} to USD {o_now:.0f} a barrel. Over the same time, "
                   f"the ringgit is {ringgit_word} against the US dollar.")
        st.plotly_chart(mini_lines(recent, ["MYR_per_USD", "Brent_USD"],
                                   ["Ringgit per US dollar", "Oil (USD)"], [RINGGIT, OIL], dual=True),
                        width="stretch")

    with t2.container(border=True, height=TAKE_H):
        recent2 = df2[df2.index >= n_start]
        d_then, d_now = recent2["USD_index"].iloc[0], recent2["USD_index"].iloc[-1]
        d_pct = (d_now / d_then - 1) * 100
        st.markdown(f"**The US dollar got {abs(d_pct):.1f}% {'stronger' if d_pct > 0 else 'weaker'} worldwide**")
        st.caption(f"Compared with other major currencies like the euro, yen and yuan, the US dollar "
                   f"got {abs(d_pct):.1f}% {'stronger' if d_pct > 0 else 'weaker'} (gold line). "
                   f"Over the same time, the ringgit is {ringgit_word} against the US dollar.")
        st.plotly_chart(mini_lines(recent2, ["MYR_per_USD", "USD_index"],
                                   ["Ringgit per US dollar", "US dollar strength"], [RINGGIT, DOLLAR], dual=True),
                        width="stretch")

    with t3.container(border=True, height=TAKE_H):
        rp = fx["USD"].dropna()
        rp = rp[rp.index >= n_start]
        best_day, worst_day = rp.idxmin(), rp.idxmax()
        st.markdown("**The ringgit's best and worst days**")
        st.caption(f"Strongest: RM{rp.min():.2f} for 1 US dollar on {best_day:%d %b %Y}. "
                   f"Weakest: RM{rp.max():.2f} on {worst_day:%d %b %Y}. Today: RM{rp.iloc[-1]:.2f}.")
        hl = go.Figure()
        hl.add_trace(go.Scatter(x=rp.index, y=rp.values, name="Ringgit per US dollar",
                                line=dict(color=RINGGIT, width=2), hovertemplate="RM%{y:.2f}"))
        hl.add_trace(go.Scatter(x=[best_day], y=[rp.min()], mode="markers+text", name="Strongest",
                                marker=dict(color=OIL, size=10), text=["Strongest"],
                                textposition="bottom center", hovertemplate="RM%{y:.2f}"))
        hl.add_trace(go.Scatter(x=[worst_day], y=[rp.max()], mode="markers+text", name="Weakest",
                                marker=dict(color=WARN, size=10), text=["Weakest"],
                                textposition="top center", hovertemplate="RM%{y:.2f}"))
        style(hl, 260)
        hl.update_layout(showlegend=False, margin=dict(l=5, r=5, t=20, b=5), font=dict(size=11))
        st.plotly_chart(hl, width="stretch")

# ===============================================================
# PAGE 2: for technical readers
# ===============================================================
with page2:
    st.caption("This page shows how the results were found, for readers who want the method.")
    mae = r["mae"]

    def sen(v):
        return f"{round(v * 100):.0f} sen"

    # 1. Data
    st.subheader("1. The data")
    st.markdown(
        "- **Ringgit:** daily price of 1 US dollar at 12pm, from Bank Negara Malaysia (data.gov.my)\n"
        "- **Oil:** daily Brent oil price in US dollars, from FRED\n"
        "- **US dollar strength:** the US dollar compared with other major currencies, from FRED\n"
        f"- **Period:** 22 July 2005 to today. I start in 2005 because before that, the ringgit was fixed "
        f"at RM3.80, so the market could not move it. After joining the data: {len(df):,} days."
    )

    # 2. Correlation
    st.subheader("2. Do oil and the ringgit move together?")
    c = r["corr"]
    a_, b_ = st.columns([1, 2])
    a_.dataframe(
        pd.DataFrame({"Period": list(c.keys()), "Correlation": [round(v, 2) for v in c.values()]}),
        hide_index=True, width="stretch",
    )
    b_.markdown(
        "**Correlation** is one number from −1 to +1 that shows how closely two things move together. "
        "−1 means they always move in opposite directions, 0 means no link.\n\n"
        f"Before 2022 it was **{c['Before 2022']:.2f}**: when oil went up, the ringgit usually got stronger. "
        f"Since 2022 it is **{c['2022 to now']:.2f}**, so the link has almost disappeared."
    )

    # 3. Models
    st.subheader("3. Can oil explain the ringgit?")
    st.markdown(
        "I built a simple **linear regression** model: a straight-line rule that guesses the price of "
        "1 US dollar from the oil price. I trained it on 2005 to 2021 (including the 2008 and 2014 oil "
        "crashes and COVID), then tested it on 2022 to now, which it had never seen. Then I built a second model that also uses the US dollar's strength."
    )
    rows = {"2005 to 2021 (training)": "Train (studied)", "2022 to now (test)": "Test: 2022–now"}
    a_, b_ = st.columns([2, 3])
    a_.markdown("**How far off the guess was, on average**")
    a_.dataframe(
        pd.DataFrame({
            "Period": list(rows),
            "Oil only": [sen(mae[k]["Oil only"]) for k in rows.values()],
            "Oil + US dollar": [sen(mae[k]["Oil + US dollar"]) for k in rows.values()],
        }),
        hide_index=True, width="stretch",
    )
    a_.markdown(
        f"Oil alone was about {sen(mae['Train (studied)']['Oil only'])} off up to 2021, but since 2022 it is about "
        f"{sen(mae['Test: 2022–now']['Oil only'])} off. With the US dollar's strength added, it is "
        f"about {sen(mae['Test: 2022–now']['Oil + US dollar'])} off."
    )
    g = r["guesses"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=g.index, y=g["MYR_per_USD"], name="Real price",
                              line=dict(color=RINGGIT, width=2), hovertemplate="RM%{y:.2f}"))
    fig2.add_trace(go.Scatter(x=g.index, y=g["Oil only"], name="Guess: oil only",
                              line=dict(color=OIL, width=1.2), hovertemplate="RM%{y:.2f}"))
    fig2.add_trace(go.Scatter(x=g.index, y=g["Oil + US dollar"], name="Guess: oil + US dollar",
                              line=dict(color=DOLLAR, width=1.2), hovertemplate="RM%{y:.2f}"))
    mark_date(fig2, TEST_START, "Test starts (2022)", WARN, y=0.99)
    style(fig2, 360)
    fig2.update_layout(margin=dict(l=5, r=5, t=20, b=5))
    fig2.update_yaxes(tickprefix="RM")
    with b_.container(border=True):
        st.plotly_chart(fig2, width="stretch")

    # 4. Prediction
    st.subheader("4. Can it predict next month?")
    f = r["forecast"]
    a_, b_, c_ = st.columns(3)
    a_.metric("Simple guess: next month = this month", f"{f['lazy'] * 100:.1f} sen off", border=True)
    b_.metric("My model (ringgit + oil + US dollar)", f"{f['model'] * 100:.1f} sen off", border=True)
    c_.markdown(
        f"Tested on {f['months']} months from 2022. My model was **not better** than simply guessing "
        "that next month's price will be the same as this month's. This is normal for exchange rates, "
        "so this project explains the past rather than predicting the future."
    )

    # Limits
    st.subheader("What this analysis cannot tell us")
    st.markdown(
        "- **Moving together does not prove one causes the other.**\n"
        "- Oil and the US dollar affect each other, so their effects cannot be fully separated.\n"
        "- Other things also move the ringgit (interest rates, politics, trade, investors) and are not included."
    )

st.divider()
st.caption("Built by aamrhh · Python, pandas, scikit-learn, Plotly, Streamlit · "
           "[github.com/aamrhh/ringgit-oil-analysis](https://github.com/aamrhh/ringgit-oil-analysis)")