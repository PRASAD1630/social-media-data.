import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Instagram AI Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   padding: 20px; border-radius: 10px; color: white; }
    h1 { color: #667eea; text-align: center; padding: 20px 0; }
    h2 { color: #764ba2; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("instagram_preprocessed.csv")
    except FileNotFoundError:
        st.error("❌ instagram_preprocessed.csv not found. Please generate it first from the notebook.")
        return None
    
    # Data preprocessing
    if "engagement_rate" not in df.columns:
        df["engagement_rate"] = (df["engagement"] / (df["likes_count"] + 1)) * 100
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed')
        df["hour"] = df["timestamp"].dt.hour
        df["date"] = df["timestamp"].dt.date
        df["day_name"] = df["timestamp"].dt.day_name()
    
    return df

df = load_data()

if df is None:
    st.stop()

# ==================== SIDEBAR ====================
st.sidebar.title("📊 Dashboard Controls")
st.sidebar.markdown("---")

# Filters
available_languages = df["language"].unique() if "language" in df.columns else []
available_media = df["media_type"].unique() if "media_type" in df.columns else []
available_locations = df["location"].unique() if "location" in df.columns else []

language_filter = st.sidebar.selectbox(
    "🌍 Select Language",
    ["All Languages"] + list(available_languages)
)

media_filter = st.sidebar.selectbox(
    "🎬 Select Media Type",
    ["All Media Types"] + list(available_media)
)

location_filter = st.sidebar.selectbox(
    "📍 Select Location",
    ["All Locations"] + list(available_locations)
)

# Apply filters
df_filtered = df.copy()
if language_filter != "All Languages":
    df_filtered = df_filtered[df_filtered["language"] == language_filter]
if media_filter != "All Media Types":
    df_filtered = df_filtered[df_filtered["media_type"] == media_filter]
if location_filter != "All Locations":
    df_filtered = df_filtered[df_filtered["location"] == location_filter]

# Navigation
st.sidebar.markdown("---")
st.sidebar.title("📑 Navigation")
page = st.sidebar.radio("Select Page", [
    "🏠 Overview",
    "📊 Engagement Analysis",
    "🔬 Advanced Analytics",
    "🎯 Performance Metrics",
    "💾 Data Export"
])

st.sidebar.markdown("---")
st.sidebar.info("📈 Instagram AI Analytics Dashboard v1.0\n\nBuilt with Streamlit | Data Science Project")

# ==================== HEADER METRICS ====================
st.title("🚀 Instagram AI Analytics Dashboard")
st.markdown(f"Filtered Data: **{len(df_filtered):,}** posts | Total Dataset: **{len(df):,}** posts")
st.markdown("---")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📝 Total Posts", f"{len(df_filtered):,}")
with col2:
    st.metric("👍 Avg Likes", f"{int(df_filtered['likes_count'].mean()):,}")
with col3:
    st.metric("💬 Avg Comments", f"{int(df_filtered['comments_count'].mean()):,}")
with col4:
    st.metric("🔥 Avg Engagement", f"{int(df_filtered['engagement'].mean()):,}")
with col5:
    st.metric("📈 Engagement Rate", f"{df_filtered['engagement_rate'].mean():.2f}%")

st.markdown("---")

# ==================== PAGE: OVERVIEW ====================
if page == "🏠 Overview":
    st.header("📊 Overview & Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    # Language Distribution
    with col1:
        lang_data = df["language"].value_counts().reset_index()
        lang_data.columns = ["language", "count"]
        fig_lang = px.bar(
            lang_data, 
            x="language", 
            y="count",
            color="count",
            title="🌍 Language Distribution",
            color_continuous_scale="Viridis",
            labels={"count": "Number of Posts", "language": "Language"}
        )
        st.plotly_chart(fig_lang, use_container_width=True)
    
    # Sentiment Distribution
    with col2:
        if "sentiment" in df_filtered.columns:
            sentiment_data = df_filtered["sentiment"].value_counts().reset_index()
            sentiment_data.columns = ["sentiment", "count"]
            fig_sentiment = px.pie(
                sentiment_data,
                names="sentiment",
                values="count",
                title="😊 Sentiment Distribution",
                color_discrete_map={
                    "positive": "#2ecc71",
                    "negative": "#e74c3c",
                    "neutral": "#95a5a6"
                }
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Media Type Distribution
    col1, col2 = st.columns(2)
    with col1:
        if "media_type" in df.columns:
            media_data = df["media_type"].value_counts().reset_index()
            media_data.columns = ["media_type", "count"]
            fig_media = px.pie(
                media_data,
                names="media_type",
                values="count",
                title="🎬 Media Type Distribution"
            )
            st.plotly_chart(fig_media, use_container_width=True)
    
    # Location Distribution
    with col2:
        location_data = df["location"].value_counts().head(10).reset_index()
        location_data.columns = ["location", "count"]
        fig_location = px.bar(
            location_data,
            y="location",
            x="count",
            orientation="h",
            title="📍 Top 10 Locations",
            color="count",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_location, use_container_width=True)
    
    # Hashtag Analysis
    st.subheader("🔖 Top Hashtags Analysis")
    all_hashtags = []
    for tags in df_filtered["hashtags"]:
        if isinstance(tags, list):
            all_hashtags.extend(tags)
    
    if all_hashtags:
        top_hashtags = Counter(all_hashtags).most_common(15)
        hashtag_df = pd.DataFrame(top_hashtags, columns=["hashtag", "count"])
        fig_hashtags = px.bar(
            hashtag_df,
            y="hashtag",
            x="count",
            orientation="h",
            title="🔥 Top 15 Hashtags",
            color="count",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_hashtags, use_container_width=True)

# ==================== PAGE: ENGAGEMENT ANALYSIS ====================
elif page == "📊 Engagement Analysis":
    st.header("📊 Engagement Analysis & Patterns")
    
    col1, col2 = st.columns(2)
    
    # Hashtags vs Engagement
    with col1:
        sample_data = df_filtered.sample(min(3000, len(df_filtered)))
        fig_hashtags_eng = px.scatter(
            sample_data,
            x="hashtag_count",
            y="engagement",
            color="language",
            size="likes_count",
            hover_data=["sentiment", "media_type"],
            title="📌 Hashtags vs Engagement",
            labels={"hashtag_count": "Number of Hashtags", "engagement": "Total Engagement"}
        )
        st.plotly_chart(fig_hashtags_eng, use_container_width=True)
    
    # Likes Distribution
    with col2:
        fig_likes = px.histogram(
            df_filtered,
            x="likes_count",
            nbins=50,
            title="👍 Likes Distribution",
            color_discrete_sequence=["#3498db"],
            labels={"likes_count": "Likes", "count": "Frequency"}
        )
        st.plotly_chart(fig_likes, use_container_width=True)
    
    # Engagement by Sentiment
    if "sentiment" in df_filtered.columns:
        st.subheader("Engagement by Sentiment")
        fig_sentiment_eng = px.box(
            df_filtered,
            x="sentiment",
            y="engagement",
            color="sentiment",
            title="📊 Engagement Distribution by Sentiment",
            color_discrete_map={
                "positive": "#2ecc71",
                "negative": "#e74c3c",
                "neutral": "#95a5a6"
            }
        )
        st.plotly_chart(fig_sentiment_eng, use_container_width=True)
    
    # Hourly Engagement Pattern
    if "hour" in df_filtered.columns:
        st.subheader("⏰ Engagement Pattern by Hour")
        hourly_data = df_filtered.groupby("hour")[["engagement", "likes_count", "comments_count"]].mean().reset_index()
        fig_hourly = px.line(
            hourly_data,
            x="hour",
            y=["engagement", "likes_count", "comments_count"],
            title="Hourly Engagement Patterns",
            markers=True,
            labels={"hour": "Hour of Day (0-23)", "value": "Average Count"}
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Caption Length vs Engagement
    st.subheader("📝 Caption Length vs Engagement")
    col1, col2 = st.columns(2)
    with col1:
        fig_caption_length = px.scatter(
            sample_data,
            x="caption_length",
            y="engagement",
            color="likes_count",
            title="Caption Length vs Engagement",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_caption_length, use_container_width=True)
    
    with col2:
        fig_word_count = px.scatter(
            sample_data,
            x="word_count",
            y="engagement",
            color="likes_count",
            title="Word Count vs Engagement",
            color_continuous_scale="Plasma"
        )
        st.plotly_chart(fig_word_count, use_container_width=True)

# ==================== PAGE: ADVANCED ANALYTICS ====================
elif page == "🔬 Advanced Analytics":
    st.header("🔬 Advanced Analytics & Insights")
    
    col1, col2 = st.columns(2)
    
    # Correlation Matrix
    with col1:
        st.subheader("📊 Correlation Matrix")
        num_cols = [col for col in ["likes_count", "comments_count", "shares_count", 
                                     "engagement", "caption_length", "word_count", 
                                     "hashtag_count", "engagement_rate"] 
                   if col in df_filtered.columns]
        
        if len(num_cols) > 1:
            corr_matrix = df_filtered[num_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale="RdBu",
                title="Correlation Matrix",
                labels=dict(color="Correlation")
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # Language Performance
    with col2:
        st.subheader("🌍 Language Performance")
        lang_perf = df.groupby("language").agg({
            "likes_count": "mean",
            "engagement": "mean",
            "engagement_rate": "mean"
        }).reset_index().sort_values("engagement", ascending=False)
        
        fig_lang_perf = px.bar(
            lang_perf,
            x="language",
            y=["likes_count", "engagement"],
            title="Average Performance by Language",
            barmode="group",
            color_discrete_sequence=["#3498db", "#e74c3c"]
        )
        st.plotly_chart(fig_lang_perf, use_container_width=True)
    
    # Media Type vs Sentiment
    if "sentiment" in df_filtered.columns:
        st.subheader("🎬 Media Type vs Sentiment Performance")
        media_sentiment = df_filtered.groupby(["media_type", "sentiment"]).agg({
            "engagement": "mean",
            "likes_count": "mean"
        }).reset_index()
        
        fig_media_sentiment = px.bar(
            media_sentiment,
            x="media_type",
            y="engagement",
            color="sentiment",
            title="Average Engagement by Media Type & Sentiment",
            barmode="group"
        )
        st.plotly_chart(fig_media_sentiment, use_container_width=True)
    
    # Distribution Comparisons
    st.subheader("📈 Distribution Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_dist_likes = px.histogram(
            df_filtered,
            x="likes_count",
            nbins=40,
            title="Likes Distribution",
            marginal="box"
        )
        st.plotly_chart(fig_dist_likes, use_container_width=True)
    
    with col2:
        fig_dist_comments = px.histogram(
            df_filtered,
            x="comments_count",
            nbins=40,
            title="Comments Distribution",
            marginal="box"
        )
        st.plotly_chart(fig_dist_comments, use_container_width=True)
    
    with col3:
        fig_dist_engagement = px.histogram(
            df_filtered,
            x="engagement",
            nbins=40,
            title="Engagement Distribution",
            marginal="box"
        )
        st.plotly_chart(fig_dist_engagement, use_container_width=True)

# ==================== PAGE: PERFORMANCE METRICS ====================
elif page == "🎯 Performance Metrics":
    st.header("🎯 Performance Metrics & ROI Analysis")
    
    # Media Type Performance
    st.subheader("🎬 Performance by Media Type")
    col1, col2 = st.columns(2)
    
    media_perf = df_filtered.groupby("media_type").agg({
        "likes_count": "mean",
        "comments_count": "mean",
        "engagement": "mean",
        "engagement_rate": "mean",
        "post_id": "count"
    }).rename(columns={"post_id": "total_posts"}).reset_index()
    
    with col1:
        fig_media_likes = px.bar(
            media_perf,
            x="media_type",
            y="likes_count",
            title="Avg Likes by Media Type",
            color="likes_count",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_media_likes, use_container_width=True)
    
    with col2:
        fig_media_engagement = px.bar(
            media_perf,
            x="media_type",
            y="engagement",
            title="Avg Engagement by Media Type",
            color="engagement",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_media_engagement, use_container_width=True)
    
    # Location Performance
    st.subheader("📍 Performance by Location")
    col1, col2 = st.columns(2)
    
    location_perf = df_filtered.groupby("location").agg({
        "likes_count": "mean",
        "engagement": "mean",
        "post_id": "count"
    }).rename(columns={"post_id": "total_posts"}).sort_values("likes_count", ascending=True).reset_index()
    
    with col1:
        fig_loc_likes = px.bar(
            location_perf,
            y="location",
            x="likes_count",
            orientation="h",
            title="Avg Likes by Location",
            color="likes_count",
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig_loc_likes, use_container_width=True)
    
    with col2:
        fig_loc_engagement = px.bar(
            location_perf,
            y="location",
            x="engagement",
            orientation="h",
            title="Avg Engagement by Location",
            color="engagement",
            color_continuous_scale="Oranges"
        )
        st.plotly_chart(fig_loc_engagement, use_container_width=True)
    
    # Performance Metrics Table
    st.subheader("📊 Detailed Performance Metrics")
    st.dataframe(
        media_perf.round(2),
        use_container_width=True,
        hide_index=True
    )

# ==================== PAGE: DATA EXPORT ====================
elif page == "💾 Data Export":
    st.header("💾 Data Export & Download")
    
    st.info("📥 Download the filtered dataset in CSV format for further analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Display Data Preview
        st.subheader("📋 Data Preview")
        st.dataframe(
            df_filtered.head(20),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.subheader("📊 Dataset Info")
        st.metric("Total Records", len(df_filtered))
        st.metric("Total Columns", len(df_filtered.columns))
        st.metric("Memory Usage", f"{df_filtered.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.subheader("📝 Column Names")
        for i, col in enumerate(df_filtered.columns, 1):
            st.text(f"{i}. {col}")
    
    st.markdown("---")
    
    # Download Button
    st.subheader("⬇️ Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_filtered = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_filtered,
            file_name="instagram_filtered_data.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_all = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Data (CSV)",
            data=csv_all,
            file_name="instagram_full_data.csv",
            mime="text/csv"
        )
    
    with col3:
        st.subheader("")
        st.write("")
        if st.button("🔄 Refresh Data Cache"):
            st.cache_data.clear()
            st.success("✅ Cache cleared! Data will reload on next interaction.")
    
    # Data Statistics
    st.markdown("---")
    st.subheader("📈 Statistical Summary")
    
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    st.dataframe(
        df_filtered[numeric_cols].describe().round(2),
        use_container_width=True
    )

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>📊 <b>Instagram AI Analytics Dashboard v1.0</b></p>
    <p>Built with Streamlit | Data Science & Machine Learning Project</p>
    <p>✨ Advanced Analytics | Real-time Insights | Performance Metrics</p>
</div>
""", unsafe_allow_html=True)
