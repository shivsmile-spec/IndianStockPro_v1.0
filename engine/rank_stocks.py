import json, math, os, re, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Free-data prototype. Do not treat this as an exchange-certified feed.
# The scoring is deliberately transparent and explainable.
UNIVERSE = [
"RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","BHARTIARTL","ITC","SBIN","LT","HINDUNILVR",
"AXISBANK","KOTAKBANK","BAJFINANCE","M&M","MARUTI","SUNPHARMA","TITAN","ADANIENT","ADANIPORTS",
"NTPC","POWERGRID","ONGC","COALINDIA","TATASTEEL","JSWSTEEL","HCLTECH","WIPRO","TECHM",
"ULTRACEMCO","ASIANPAINT","NESTLEIND","BAJAJFINSV","HINDALCO","TRENT","BEL","HAL","INDUSINDBK",
"DRREDDY","CIPLA","EICHERMOT","TATAMOTORS","TATA consumer","GRASIM","DIVISLAB","APOLLOHOSP",
"BRITANNIA","HEROMOTOCO","BAJAJ-AUTO","TVSMOTOR","SIEMENS","ABB","DLF","INDIGO","VEDL"
]
# Yahoo Finance NSE suffixes.
TICKERS = {s: (s.replace(" ","") + ".NS") for s in UNIVERSE}
BENCHMARK = "^NSEI"

def rsi(series, n=14):
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100/(1+rs))

def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([(df["High"]-df["Low"]), (df["High"]-prev).abs(), (df["Low"]-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def clamp(x, lo=0, hi=100):
    if x is None or not np.isfinite(x): return 50.0
    return float(max(lo, min(hi, x)))

def pct_score(x, lo, hi):
    return clamp((x-lo)/(hi-lo)*100)

def company_name(symbol):
    # Keep the app usable even when Yahoo metadata is incomplete.
    return symbol.replace("_"," ").title()

def download(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True,
                         progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if len(df) < 220: return None
        return df
    except Exception as e:
        print("download failed", ticker, e)
        return None

def factors(df, bench):
    close = df["Close"]; vol = df["Volume"]
    price = float(close.iloc[-1])
    r5 = float(close.pct_change(5).iloc[-1]*100)
    r20 = float(close.pct_change(20).iloc[-1]*100)
    r60 = float(close.pct_change(60).iloc[-1]*100)
    rsi14 = float(rsi(close).iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    vol20 = float(vol.rolling(20).mean().iloc[-1])
    vol_ratio = float(vol.iloc[-1]/vol20) if vol20 else 1
    high20 = float(close.rolling(20).max().iloc[-1])
    dist_high = float((price/high20-1)*100)
    atr14 = float(atr(df).iloc[-1])
    atr_pct = atr14/price*100 if price else 99

    b = bench.reindex(close.index).ffill().dropna()
    c = close.reindex(b.index)
    rel5 = float((c.pct_change(5).iloc[-1]-b.pct_change(5).iloc[-1])*100)
    rel20 = float((c.pct_change(20).iloc[-1]-b.pct_change(20).iloc[-1])*100)
    rel60 = float((c.pct_change(60).iloc[-1]-b.pct_change(60).iloc[-1])*100)

    # Transparent factor scores.
    momentum = 0.25*pct_score(r5,-8,8)+0.45*pct_score(r20,-12,15)+0.30*pct_score(r60,-20,30)
    trend = 35*(price>sma20)+35*(sma20>sma50)+30*(price>sma200)
    rs = 0.30*pct_score(rel5,-5,5)+0.45*pct_score(rel20,-8,8)+0.25*pct_score(rel60,-12,12)
    volume = pct_score(vol_ratio,0.7,2.5)
    rsi_score = 100 - abs(rsi14-58)*2.2
    rsi_score = clamp(rsi_score)
    breakout = pct_score(dist_high,-12,0)
    volatility = clamp(100 - atr_pct*10)
    # Risk/reward proxy: target at recent high + ATR, invalidation below swing/ATR.
    entry = price
    stop = max(0.01, price - 1.5*atr14)
    target = max(price + 2.5*atr14, high20)
    rr = (target-price)/(price-stop) if price>stop else 0
    rr_score = pct_score(rr,0.8,3.5)

    total = (0.22*momentum + 0.18*trend + 0.16*rs + 0.10*volume +
             0.08*rsi_score + 0.08*breakout + 0.08*volatility + 0.10*rr_score)
    score = clamp(total)

    reasons=[]
    if r20>3: reasons.append(f"20-day momentum is +{r20:.1f}%.")
    if price>sma20>sma50 and price>sma200: reasons.append("Price is above 20/50/200-day trend levels.")
    if rel20>1.5: reasons.append(f"Relative strength vs NIFTY is +{rel20:.1f}% over 20 days.")
    if vol_ratio>1.25: reasons.append(f"Volume is {vol_ratio:.1f}× its 20-day average.")
    if 48<=rsi14<=68: reasons.append(f"RSI 14 is {rsi14:.1f}, avoiding an extreme overbought reading.")
    if dist_high>-3: reasons.append(f"Price is within {abs(dist_high):.1f}% of its 20-day high.")
    if rr>=2: reasons.append(f"ATR-based risk/reward proxy is about {rr:.1f}:1.")
    if not reasons: reasons.append("No single factor dominates; ranking comes from the combined score.")

    risk = clamp(100-volatility)
    confidence = clamp(0.7*score + 0.3*(100-abs(rsi14-55)*1.5))
    signal = "Strong setup" if score>=78 else "Positive setup" if score>=68 else "Watch" if score>=55 else "Avoid/weak"

    return {
        "price": round(price,2), "score": round(score,1), "signal": signal,
        "opportunity": round(score,1), "confidence": round(confidence,1), "risk": round(risk,1),
        "horizon":"1–4 weeks",
        "entry":round(entry,2),"target":round(target,2),"stop":round(stop,2),
        "factors":{
            "Momentum":round(momentum,1),"Trend":round(trend,1),"Relative strength":round(rs,1),
            "Volume":round(volume,1),"RSI quality":round(rsi_score,1),
            "Breakout proximity":round(breakout,1),"Volatility":round(volatility,1),"Risk/reward":round(rr_score,1)
        },
        "raw":{"rsi14":round(rsi14,2),"return5d":round(r5,2),"return20d":round(r20,2),
               "return60d":round(r60,2),"rel20d":round(rel20,2),"volumeRatio":round(vol_ratio,2),
               "atrPct":round(atr_pct,2),"distance20dHighPct":round(dist_high,2),"rrProxy":round(rr,2)},
        "reasons":reasons,
        "catalysts":["Momentum/trend/volume setup detected; verify current company news before acting."],
        "risks":["Model does not know future events; earnings, gaps, macro shocks and liquidity can invalidate the setup."]
    }

def main():
    print("Downloading benchmark...")
    bench = download(BENCHMARK)
    if bench is None:
        raise RuntimeError("Could not download NIFTY benchmark data.")
    bench_close = bench["Close"]
    rows=[]
    stocks=[]
    for i,(sym,ticker) in enumerate(TICKERS.items(),1):
        print(f"[{i}/{len(TICKERS)}] {sym}")
        df=download(ticker)
        if df is None: continue
        try:
            x=factors(df,bench_close)
            x["symbol"]=sym
            x["company"]=company_name(sym)
            rows.append(x)
            stocks.append({"symbol":sym,"company":company_name(sym)})
        except Exception as e:
            print("factor error",sym,e)
        time.sleep(0.15)

    rows.sort(key=lambda x:x["score"], reverse=True)
    # Keep the output compact for the mobile app.
    result={
      "version":"1.0",
      "generated":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
      "market":{"regime":"NIFTY benchmark loaded","benchmark":"^NSEI"},
      "methodology":{
        "weights":{"momentum":22,"trend":18,"relativeStrength":16,"volume":10,"rsiQuality":8,
                   "breakout":8,"volatility":8,"riskReward":10},
        "note":"Scores are research signals, not guaranteed predictions or investment advice."
      },
      "rankings":rows[:10],
      "stocks":stocks
    }
    out=os.path.join(DATA_DIR,"rankings.json")
    with open(out,"w",encoding="utf-8") as f: json.dump(result,f,indent=2,ensure_ascii=False)
    print("Wrote",out)

if __name__=="__main__":
    main()
