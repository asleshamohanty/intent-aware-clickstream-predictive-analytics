import pandas as pd
import numpy as np

# Define funnel stages
FUNNEL = ["Landing", "Product", "Cart", "Checkout", "Purchase"]

def compute_behavior_features(input_csv, output_csv="data/processed/synthetic_clickstream_final.csv"):
    """
    Compute behavioral features from clickstream:
    - Click count
    - Average dwell time
    - Max funnel stage reached
    - Session duration
    - One-hot encode device
    """
    df = pd.read_csv(input_csv)

    # Ensure Timestamp is datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    session_features = []

    for session_id, group in df.groupby('SessionID'):
        group = group.sort_values('Timestamp')
        
        # Click count
        click_count = group.shape[0]

        # Average dwell time
        dwell_times = group['Timestamp'].diff().dt.total_seconds().iloc[1:]  # skip first NaN
        avg_dwell_time = dwell_times.mean() if not dwell_times.empty else 0

        # Max funnel stage reached
        max_stage_idx = max([FUNNEL.index(p) for p in group['Page'] if p in FUNNEL], default=-1)
        max_funnel_stage = FUNNEL[max_stage_idx] if max_stage_idx >= 0 else "None"

        # Session duration
        session_duration = (group['Timestamp'].max() - group['Timestamp'].min()).total_seconds()

        # Device (take the first device in session)
        device = group['Device'].iloc[0]
        device_onehot = {
            'device_mobile': int(device == 'mobile'),
            'device_desktop': int(device == 'desktop'),
            'device_tablet': int(device == 'tablet')
        }

        # Labels / NLP features (take first row)
        converted = group['Converted'].iloc[0]
        sentiment = group['Sentiment'].iloc[0]
        keywords = group['Keywords'].iloc[0]

        # TF-IDF features
        tfidf_cols = [c for c in df.columns if c.startswith('tfidf_')]
        tfidf_values = group.iloc[0][tfidf_cols].to_dict()

        # Combine all features
        features = {
            'SessionID': session_id,
            'click_count': click_count,
            'avg_dwell_time': avg_dwell_time,
            'max_funnel_stage': max_funnel_stage,
            'session_duration': session_duration,
            'Converted': converted,
            'Sentiment': sentiment,
            'Keywords': keywords
        }
        features.update(device_onehot)
        features.update(tfidf_values)

        session_features.append(features)

    final_df = pd.DataFrame(session_features)
    final_df.to_csv(output_csv, index=False)
    print(f"Final behavioral + NLP features dataset saved to {output_csv}")
    return final_df

# Example usage
if __name__ == "__main__":
    compute_behavior_features("data/processed/synthetic_clickstream_nlp_features.csv")
