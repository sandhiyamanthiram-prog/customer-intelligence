import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

os.makedirs("outputs", exist_ok=True)

# =================================
# Build RFM
# =================================

orders = pd.read_csv("data/clean/orders.csv")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"]
)

completed = orders[
    orders["status"] == "completed"
]

snapshot = pd.Timestamp(
    "2025-10-01 00:00:00"
)

rfm = (
    completed
    .groupby("customer_id")
    .agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    )
)

rfm["recency"] = (
    snapshot - rfm["last_order"]
).dt.days

rfm = rfm.reset_index()

# =================================
# Scaling
# =================================

features = rfm[
    ["recency", "frequency", "monetary"]
]

scaler = StandardScaler()

X = scaler.fit_transform(
    features
)

# =================================
# K Evaluation
# =================================

print("\nK-Means Evaluation")

results = []

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X)

    inertia = model.inertia_

    sil = silhouette_score(
        X,
        labels
    )

    results.append(
        [k, inertia, sil]
    )

    print(
        f"K={k} "
        f"Inertia={inertia:.1f} "
        f"Silhouette={sil:.3f}"
    )

# =================================
# Final Model
# =================================

final_k = 4

kmeans = KMeans(
    n_clusters=final_k,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = (
    kmeans.fit_predict(X)
)

# =================================
# Profile Segments
# =================================

profile = (
    rfm
    .groupby("segment_id")
    [
        ["recency",
         "frequency",
         "monetary"]
    ]
    .mean()
)

sizes = (
    rfm["segment_id"]
    .value_counts()
    .sort_index()
)

print("\nSegment Sizes")
print(sizes)

print("\nSegment Profile")
print(profile)

# =================================
# Premium Segment
# =================================

premium_cluster = (
    profile["monetary"]
    .idxmax()
)

rfm["is_premium"] = (
    rfm["segment_id"]
    == premium_cluster
)

segment_names = {}

for seg in profile.index:

    if seg == premium_cluster:

        segment_names[seg] = "Premium"

    elif profile.loc[seg,
         "frequency"] > 15:

        segment_names[seg] = "Loyal"

    elif profile.loc[seg,
         "recency"] > 50:

        segment_names[seg] = "At Risk"

    else:

        segment_names[seg] = "Occasional"

rfm["segment_name"] = (
    rfm["segment_id"]
    .map(segment_names)
)

# =================================
# PCA
# =================================

pca = PCA(
    n_components=2
)

pca.fit(X)

print("\nPCA Explained Variance")

for i, val in enumerate(
        pca.explained_variance_ratio_,
        start=1):

    print(
        f"PC{i}: {val:.3f}"
    )

print(
    "Total:",
    round(
        pca.explained_variance_ratio_.sum(),
        3
    )
)

# =================================
# Save
# =================================

rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
].to_csv(
    "outputs/customer_segments.csv",
    index=False
)

print(
    "\ncustomer_segments.csv saved"
)