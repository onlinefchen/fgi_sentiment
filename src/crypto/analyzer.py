# src/analyzer.py
from typing import List, Dict
import pandas as pd

def calculate_correlation(data: List[Dict], days: int = None) -> Dict:
    """
    Calculate correlation between FGI and BTC price.

    Returns:
        Dict with pearson and spearman correlation coefficients
    """
    df = pd.DataFrame(data)

    if days:
        df = df.tail(days)

    pearson = df["fgi_value"].corr(df["btc_price"], method="pearson")
    spearman = df["fgi_value"].corr(df["btc_price"], method="spearman")

    return {
        "pearson": round(pearson, 3),
        "spearman": round(spearman, 3),
        "days": days or len(df)
    }
