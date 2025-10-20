import numpy as np
import pandas as pd
import datetime as dt
import random

np.random.seed(42)

# --------------------------
# Step 1: Define Pages, Devices, Queries
# --------------------------
pages = ["Landing", "Category", "Product", "Cart", "Checkout", "Purchase", "Help", "Search"]
devices = ["mobile", "desktop", "tablet"]

# Example search queries per product category
query_bank = {
    "Shoes": ["running shoes men", "best sports shoes", "cheap sneakers", "nike air max", "adidas ultraboost"],
    "Phones": ["budget smartphone", "latest iPhone", "android phone under 15000", "Samsung Galaxy deals"],
    "Laptops": ["gaming laptop RTX", "best laptop for college", "lightweight ultrabook", "macbook pro 2024"],
    "Skincare": ["face cleanser", "moisturizer for dry skin", "best sunscreen", "acne treatment"],
    "Headphones": ["noise cancelling headphones", "bluetooth earbuds", "cheap headphones", "Sony WH-1000XM5"]
}

categories = list(query_bank.keys())

# --------------------------
# Step 2: Define Page Transition Probabilities
# --------------------------
next_probs = {
    "Landing": {"Category":0.35, "Search":0.20, "Product":0.15, "Help":0.05, "Landing":0.05, "Exit":0.20},
    "Category": {"Product":0.45, "Search":0.15, "Landing":0.05, "Help":0.05, "Exit":0.30},
    "Search": {"Product":0.40, "Category":0.15, "Landing":0.10, "Help":0.05, "Exit":0.30},
    "Product": {"Cart":0.35, "Category":0.15, "Search":0.10, "Help":0.05, "Exit":0.35},
    "Cart": {"Checkout":0.55, "Product":0.10, "Exit":0.35},
    "Checkout": {"Purchase":0.55, "Cart":0.10, "Exit":0.35},
    "Purchase": {"Exit":1.0},
    "Help": {"Landing":0.20, "Search":0.20, "Exit":0.60}
}

def sample_next(curr):
    choices, probs = zip(*next_probs[curr].items())
    return np.random.choice(choices, p=probs)

# --------------------------
# Step 3: Simulate a Single Session
# --------------------------
def simulate_session(session_id, start_time):
    device = np.random.choice(devices, p=[0.55,0.35,0.10])
    curr = "Landing"
    t = start_time
    path = []
    while True:
        path.append((session_id, t, curr, device))
        nxt = sample_next(curr)
        if nxt == "Exit":
            break
        t += dt.timedelta(seconds=np.random.randint(5,60))
        curr = nxt
    return path

# --------------------------
# Step 4: Generate Multiple Sessions
# --------------------------
sessions = []
base_time = dt.datetime(2024,10,1,8,0,0)
num_sessions = 500  # You can increase for bigger dataset

for s in range(num_sessions):
    start = base_time + dt.timedelta(minutes=np.random.randint(0, 1440))  # random time in a day
    sessions.extend(simulate_session(f"S{s:04d}", start))

df_clicks = pd.DataFrame(sessions, columns=["SessionID","Timestamp","Page","Device"])

# --------------------------
# Step 5: Assign NLP Search Queries
# --------------------------
search_queries = []
converted = []

for session_id in df_clicks['SessionID'].unique():
    cat = random.choice(categories)
    query = random.choice(query_bank[cat])
    # Conversion label: simple random but can be weighted based on paths
    conv = 1 if random.random() < 0.3 else 0  # 30% conversion probability
    search_queries.append((session_id, query, cat, conv))

df_queries = pd.DataFrame(search_queries, columns=["SessionID","SearchQuery","Category","Converted"])

# --------------------------
# Step 6: Merge Clickstream + Queries
# --------------------------
df_final = pd.merge(df_clicks, df_queries, on="SessionID", how="left")

# --------------------------
# Step 7: Save CSV
# --------------------------
df_final.to_csv("data/raw/synthetic_clickstream_nlp.csv", index=False)
print("Synthetic dataset saved as data/raw/synthetic_clickstream_nlp.csv")
print(df_final.head())
