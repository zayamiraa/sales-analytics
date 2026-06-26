"""
Sales Performance Analysis
===========================
End-to-end data analysis pipeline covering:
  - Data loading & validation
  - Data cleaning (missing values, type coercion)
  - Exploratory data analysis (EDA)
  - Revenue trend analysis
  - Category & regional breakdowns
  - Channel performance comparison
  - Chart exports

Author: GitHub Portfolio Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'

CHARTS_DIR = 'charts'
os.makedirs(CHARTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[LOAD] {len(df):,} rows × {df.shape[1]} columns loaded from '{path}'")
    return df


# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[CLEAN] Missing values before cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Fill missing discount_rate with median (reasonable imputation)
    median_discount = df['discount_rate'].median()
    df['discount_rate'] = df['discount_rate'].fillna(median_discount)

    # Recalculate missing revenue from units, price, and discount
    mask = df['total_revenue'].isnull()
    df.loc[mask, 'total_revenue'] = (
        df.loc[mask, 'units_sold'] *
        df.loc[mask, 'unit_price'] *
        (1 - df.loc[mask, 'discount_rate'])
    ).round(2)

    # Build a proper date column (first of each month)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))

    # Cast types
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)

    # Remove any duplicate order IDs
    before = len(df)
    df = df.drop_duplicates(subset='order_id')
    removed = before - len(df)
    if removed:
        print(f"[CLEAN] Removed {removed} duplicate order IDs.")

    print(f"[CLEAN] Missing values after cleaning: {df.isnull().sum().sum()}")
    print(f"[CLEAN] Final shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 3. SUMMARY STATISTICS
# ─────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    total_rev = df['total_revenue'].sum()
    total_units = df['units_sold'].sum()
    avg_order = df['total_revenue'].mean()
    avg_discount = df['discount_rate'].mean()

    print(f"  Total Revenue   : ${total_rev:>12,.2f}")
    print(f"  Total Units Sold: {total_units:>12,}")
    print(f"  Avg Order Value : ${avg_order:>12,.2f}")
    print(f"  Avg Discount    : {avg_discount*100:>11.1f}%")
    print(f"  Date Range      : {df['date'].min().strftime('%b %Y')} → {df['date'].max().strftime('%b %Y')}")

    print("\n  Revenue by Category:")
    cat_rev = df.groupby('category')['total_revenue'].sum().sort_values(ascending=False)
    for cat, rev in cat_rev.items():
        pct = rev / total_rev * 100
        print(f"    {cat:<18}: ${rev:>10,.0f}  ({pct:.1f}%)")

    print("\n  Revenue by Region:")
    reg_rev = df.groupby('region')['total_revenue'].sum().sort_values(ascending=False)
    for reg, rev in reg_rev.items():
        pct = rev / total_rev * 100
        print(f"    {reg:<12}: ${rev:>10,.0f}  ({pct:.1f}%)")


# ─────────────────────────────────────────────
# 4. CHARTS
# ─────────────────────────────────────────────
def chart_monthly_trend(df: pd.DataFrame):
    monthly = (
        df.groupby(['date', 'year'])['total_revenue']
        .sum()
        .reset_index()
        .sort_values('date')
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for yr, grp in monthly.groupby('year'):
        ax.plot(grp['date'], grp['total_revenue'], marker='o', linewidth=2, label=str(yr))

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.set_title('Monthly Revenue Trend by Year', fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Revenue')
    ax.legend(title='Year')
    plt.tight_layout()
    path = f'{CHARTS_DIR}/01_monthly_trend.png'
    plt.savefig(path)
    plt.close()
    print(f"[CHART] Saved → {path}")


def chart_category_revenue(df: pd.DataFrame):
    cat_rev = df.groupby('category')['total_revenue'].sum().sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(cat_rev.index, cat_rev.values, color=sns.color_palette('muted', len(cat_rev)))
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
    ax.set_title('Total Revenue by Product Category', fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('Total Revenue')

    for bar, val in zip(bars, cat_rev.values):
        ax.text(val + 500, bar.get_y() + bar.get_height()/2,
                f'${val:,.0f}', va='center', fontsize=9)

    plt.tight_layout()
    path = f'{CHARTS_DIR}/02_category_revenue.png'
    plt.savefig(path)
    plt.close()
    print(f"[CHART] Saved → {path}")


def chart_region_channel(df: pd.DataFrame):
    pivot = df.pivot_table(
        values='total_revenue', index='region',
        columns='channel', aggfunc='sum'
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind='bar', ax=ax, colormap='Set2', width=0.7)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
    ax.set_title('Revenue by Region and Sales Channel', fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('Region')
    ax.set_ylabel('Total Revenue')
    ax.legend(title='Channel', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = f'{CHARTS_DIR}/03_region_channel.png'
    plt.savefig(path)
    plt.close()
    print(f"[CHART] Saved → {path}")


def chart_discount_vs_revenue(df: pd.DataFrame):
    sample = df.sample(400, random_state=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        sample['discount_rate'] * 100,
        sample['total_revenue'],
        c=sample['units_sold'], cmap='viridis',
        alpha=0.6, edgecolors='none', s=50
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Units Sold')

    # Trend line
    z = np.polyfit(sample['discount_rate'], sample['total_revenue'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample['discount_rate'].min(), sample['discount_rate'].max(), 100)
    ax.plot(x_line * 100, p(x_line), 'r--', linewidth=1.5, label='Trend')

    ax.set_title('Discount Rate vs. Revenue (colored by Units Sold)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Discount Rate (%)')
    ax.set_ylabel('Total Revenue ($)')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend()
    plt.tight_layout()
    path = f'{CHARTS_DIR}/04_discount_vs_revenue.png'
    plt.savefig(path)
    plt.close()
    print(f"[CHART] Saved → {path}")


def chart_yoy_growth(df: pd.DataFrame):
    yearly = df.groupby('year')['total_revenue'].sum()
    growth = yearly.pct_change() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Revenue bars
    colors = ['#4C72B0', '#DD8452', '#55A868']
    ax1.bar(yearly.index.astype(str), yearly.values, color=colors)
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
    ax1.set_title('Annual Revenue', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Total Revenue')
    for i, (yr, val) in enumerate(yearly.items()):
        ax1.text(i, val + 500, f'${val:,.0f}', ha='center', fontsize=9)

    # YoY growth
    growth_vals = growth.dropna()
    bar_colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in growth_vals.values]
    ax2.bar(growth_vals.index.astype(str), growth_vals.values, color=bar_colors)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_title('Year-over-Year Revenue Growth (%)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Growth (%)')
    for i, (yr, val) in enumerate(growth_vals.items()):
        ax2.text(i, val + 0.3, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = f'{CHARTS_DIR}/05_yoy_growth.png'
    plt.savefig(path)
    plt.close()
    print(f"[CHART] Saved → {path}")


# ─────────────────────────────────────────────
# 5. EXPORT CLEAN DATA
# ─────────────────────────────────────────────
def export_summary(df: pd.DataFrame):
    summary = df.groupby(['year', 'category', 'region', 'channel']).agg(
        total_revenue=('total_revenue', 'sum'),
        total_units=('units_sold', 'sum'),
        avg_discount=('discount_rate', 'mean'),
        num_orders=('order_id', 'count')
    ).reset_index()
    summary['avg_discount'] = (summary['avg_discount'] * 100).round(2)
    summary['total_revenue'] = summary['total_revenue'].round(2)
    out_path = 'data/sales_summary.csv'
    summary.to_csv(out_path, index=False)
    print(f"\n[EXPORT] Summary table saved → {out_path} ({len(summary)} rows)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    df_raw = load_data('data/sales_data.csv')
    df = clean_data(df_raw)
    print_summary(df)

    print("\n[CHARTS] Generating visualizations...")
    chart_monthly_trend(df)
    chart_category_revenue(df)
    chart_region_channel(df)
    chart_discount_vs_revenue(df)
    chart_yoy_growth(df)

    export_summary(df)

    print("\n✅  Analysis complete. See /charts for all plots.")
