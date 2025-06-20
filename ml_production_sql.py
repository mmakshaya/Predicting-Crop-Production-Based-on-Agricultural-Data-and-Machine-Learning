
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
import pymysql
import math
from sklearn.tree import DecisionTreeRegressor
import pickle

# --- Database Connection ---
def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="crop"
    )

# --- Query Execution ---
def execute_query(query, params=None):
    conn = get_connection()
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df

df = pd.read_csv('production.csv')

# --- Page Configuration ---
st.set_page_config(
    page_title="Predicting Crop Production Based on Agricultural Data",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🚜 Smart Farming Dashboard")

#---Sidebar Navigation----
page = st.sidebar.radio(
    "Select Dataset",
    ["🏠 Home", "🌍 Geographic Insights", "📈Time-Series Trends","📊Crop Performance Overview","👨🏾‍💻 Production Predictor","🧾Detailed Report"]
)

# --- Home Page ---
if page == "🏠 Home":
    st.title("🌱Predicting Crop Production Based on Agricultural Data")

    col1, col2, col3 = st.columns(3)

    col1.metric("🌾 Highest Yielding Crop", "Cucumbers", "22,000 kg/ha")
    col2.metric("🌍 Top Producing Country", "India", "Over 100M tons")
    col3.metric("📉 Lowest Yielding Crop", "Cloves", "4,200 kg/ha")

    df = df.rename(columns={
    'Production (t)': 'Production',
    'Area harvested (ha)': 'Area',
    'Yield (kg/ha)': 'Yield'
    })

    col4, col5, col6 = st.columns(3)
    col4.metric("🌾 Total Global Production", "62.4B tons")
    col5.metric("🌍 Total Harvested Area", "8.5B ha")
    col6.metric("📊 Avg Global Yield", "7,329 kg/ha")

    st.markdown("### 🔮 Try the Crop Yield Predictor")
    st.markdown("Use the sidebar to input a crop, country, and area to predict yield efficiency instantly.")
    
    

elif page == "👨🏾‍💻 Production Predictor":
    st.title("🌾 Crop Production Predictor")
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('random_forest_tree_model.pkl', 'rb') as file:
        model= pickle.load(file)
   # Input fields
    area_code = st.number_input("Area Code (M49)",4)
    element_code = st.number_input("Element Code",5312)
    item_code = st.number_input("Item Code (CPC)",111)
    #year = st.number_input("Year",2023)
    area_harvested = st.number_input("Area Harvested (ha)",3400)
    yield_kg_per_ha = st.number_input("Yield (kg/ha)",5892.2)

# Predict
    if st.button("Predict Production (t)"):
       # Separate inputs
        unscaled_features = np.array([[area_code, element_code, item_code]])
        to_scale = np.array([[area_harvested, yield_kg_per_ha]])

        # Apply scaler
        scaled_features = scaler.transform(to_scale)

        # Combine both
        final_input = np.concatenate([unscaled_features, scaled_features], axis=1)

        # Show inputs
        st.write("🔍 Final input to model:", final_input)
        
        # Make prediction
        log_pred = model.predict(final_input)
        pred = np.expm1(log_pred[0])  # Reverse log1p
        pred = max(pred, 0)           # Clamp to zero if negative

        # Show result
        st.success(f"🌾 Predicted Production: {pred:,.2f} tonnes")



        #raw_input = np.array([[area_code, element_code, item_code, year,
                            #area_harvested,yield_kg_per_ha]])
        #st.write("🔍 Raw input:", raw_input)
        #scaled_input = scaler.transform(raw_input)
        #st.write("📊 After scaling:", scaled_input)
        #pred = model.predict(scaled_input)
        #st.success(f"🌾 Predicted Production: {pred[0]:,.2f} tonnes")
        #log_pred = model.predict(raw_input)
        #pred = np.expm1(log_pred[0])  # Reverse log1p transformation
        #pred = max(pred, 0)           # Clamp to zero
        #st.success(f" Predicted Production: {pred:,.2f} tonnes")


elif page == "📊Crop Performance Overview":
    st.header("🌱 Crop-Level Performance Analysis")
    st.subheader("🌱 Identifying Reliable High-Yield Crops Using Yield and Standard Deviation")
    query_high = """
    SELECT 
        Item,
        ROUND(AVG(Yield), 2) AS avg_yield,
        ROUND(STDDEV(Yield), 2) AS yield_stddev
    FROM crop_data
    GROUP BY Item
    ORDER BY avg_yield DESC;
    """
    df_high = execute_query(query_high)
    fig = px.bar(
      df_high,
      x='Item',
      y='avg_yield',
      color='yield_stddev',
      color_continuous_scale='Bluered',
      title="Average Yield by Crop (Colored by Std Dev)",
      labels={'avg_yield': 'Average Yield (kg/ha)', 'yield_stddev': 'Yield Std Dev'}
    )
    fig.update_layout(xaxis_tickangle=-45, height=600)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Show Data Table"):
      st.dataframe(df_high)

    st.subheader("🌾 Crops Using Large Land Area but Yielding Less")

    query = """
      SELECT 
        Item,
        SUM(Area_harvested) AS total_area,
        AVG(Yield) AS avg_yield
      FROM crop_data
      WHERE Area_harvested IS NOT NULL AND Yield IS NOT NULL
      GROUP BY Item;
      """

    df_grouped = execute_query(query)
    area_threshold = 2000000  # 2 million ha
    yield_threshold = 5000    # 5,000 kg/ha
    df_grouped["Low_Efficiency"] = (
      (df_grouped["total_area"] > area_threshold) & 
      (df_grouped["avg_yield"] < yield_threshold)
    )

# --- Plot using Plotly ---
    fig = px.scatter(
      df_grouped,
      x="total_area",
      y="avg_yield",
      color="Low_Efficiency",
      hover_name="Item",
      title="Crops Consuming Large Area but Yielding Less",
      labels={
         "total_area": "Total Area Harvested (ha)",
         "avg_yield": "Average Yield (kg/ha)"
        }
    )

# Threshold reference lines
    fig.add_vline(x=area_threshold, line_dash="dash", line_color="gray")
    fig.add_hline(y=yield_threshold, line_dash="dash", line_color="gray")

# Layout tweaks
    fig.update_traces(marker=dict(size=12, opacity=0.8))
    fig.update_layout(xaxis_tickformat=".2s", height=650)

# --- Display in Streamlit ---
    st.plotly_chart(fig, use_container_width=True)

# Optional table
    with st.expander("📊 Show Data Table"):
      st.dataframe(df_grouped.sort_values(by="total_area", ascending=False)) 

    st.subheader("🌾 Yield per Hectare by Crop")
    query = """
        SELECT Item, AVG(Yield) AS Yield
        FROM crop_data
        GROUP BY Item
        ORDER BY Yield DESC
    """
    df_grouped = execute_query(query)

    # Group items into 8 categories
    unique_items = df_grouped["Item"].unique()
    total_items = len(unique_items)
    items_per_group = math.ceil(total_items / 8)
    grouped_items = [unique_items[i:i + items_per_group] for i in range(0, total_items, items_per_group)]

    group_labels = [f"Group {i+1}" for i in range(len(grouped_items))]
    selected_group = st.radio("Select Crop Group", group_labels, horizontal=True)

    group_index = group_labels.index(selected_group)
    selected_items = grouped_items[group_index]
    filtered_df = df_grouped[df_grouped["Item"].isin(selected_items)]

    fig = px.bar(
        filtered_df.sort_values(by="Yield", ascending=False),
        x="Item",
        y="Yield",
        title=f"Average Yield per Hectare - {selected_group}",
        labels={"Item": "Crop", "Yield": "Yield (kg/ha)"},
        color="Yield",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🕵🏻Insights:")
    st.markdown("<p style='color: brown;'>The visual analyses of crop yield patterns reveal important insights into agricultural efficiency and strategic planning. The grouped bar charts demonstrate a clear yield gradient from Group 1 to Group 8, with Group 1 crops like cucumbers, sugar cane, and tomatoes producing over 20,000 kg/ha—making them ideal for intensive, high-return farming. In contrast, lower-yielding crops such as lentils, castor oil seeds, cloves, and vanilla, found in Groups 7 and 8, yield around 4,200–4,300 kg/ha but often serve niche or high-value markets. The first scatter plot further highlights a concerning trend where certain crops occupy vast land areas yet deliver subpar yields (below 5,000 kg/ha), classifying them as inefficient and signaling a need for agronomic improvement, targeted research, and policy intervention. Meanwhile, the second plot reveals that although crops like cucumbers and watermelons achieve high yields, their large standard deviations indicate yield inconsistency across regions or seasons. In contrast, staple crops such as maize, soybeans, and cotton maintain more stable yield patterns, suggesting greater resilience and predictability. Together, these findings emphasize the need to balance land use efficiency, yield consistency, and market value when planning agricultural production strategies.</p>", unsafe_allow_html=True)


elif page == "📈Time-Series Trends":  
    st.header("🗓️ Temporal Analysis (Time Series Trends)")  
    st.subheader("📈 Annual Trends for All Crops")
    query_trend = """
        SELECT 
            Year,
            Item,
            SUM(Area_harvested) AS `Total_area (ha)`,
            SUM(Production) AS `Total_production (t)`,
            SUM(Yield) AS `Total_yield (kg/ha)`,
            AVG(Yield) AS `Avg_yield (kg/ha)`
        FROM crop_data
        GROUP BY Year, Item
        ORDER BY Year;
    """
    df_grouped_trend = execute_query(query_trend)
    metric = st.selectbox("Select metric to plot", ['Total_area (ha)', 'Total_production (t)', 'Avg_yield (kg/ha)'])


    grouped_metric1 = df_grouped_trend.groupby("Item")[metric].mean().sort_values(ascending=False)

    total_crops = len(grouped_metric1)
    window_size = 10

    # Adjust slider max so that window doesn't overflow the available crops
    max_start = max(1, total_crops - window_size + 1)

    # Slider to select starting rank
    n = st.slider("Select starting rank (N)", 1, max_start, 1)

    # Select the window of crops from rank n to n+window_size-1
    start_idx = n - 1  # zero-based
    end_idx = start_idx + window_size

    selected_items = grouped_metric1.iloc[start_idx:end_idx].index

    # Filter main data to only selected crops
    filtered_df = df_grouped_trend[df_grouped_trend['Item'].isin(selected_items)]

    # Plot
    fig = px.line(
        filtered_df,
        x='Year',
        y=metric,
        color='Item',
        title=f'Annual {metric.replace("_", " ").title()} - Crops ranked {n} to {end_idx}',
        labels={'Year': 'Year', metric: metric.replace("_", " ").title(), 'Item': 'Crop'},
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# Convert Year column to datetime and extract year as integer

    df_grouped_trend['Year'] = pd.to_datetime(df_grouped_trend['Year'], dayfirst=True).dt.year

# Aggregate data by Year (sum for Area_harvested and Production, average for Yield)
    df_yearly = df_grouped_trend.groupby('Year').agg({
        'Total_area (ha)': 'sum',
        'Total_production (t)': 'sum',
        'Total_yield (kg/ha)': 'sum'
    }).reset_index()

# Melt the DataFrame to long format for Plotly Express
    df_melted = df_yearly.melt(id_vars='Year', value_vars=['Total_area (ha)', 'Total_production (t)', 'Total_yield (kg/ha)'],
                          var_name='Metric', value_name='Value')

# Plot clustered bar chart
    fig = px.bar(
      df_melted,
      x='Year',
      y='Value',
      color='Metric',
      barmode='group',
      title='Clustered Bar Chart of Area Harvested, Production and Yield by Year',
      labels={'Value': 'Value', 'Year': 'Year', 'Metric': 'Metric'}
    )

    fig.update_layout(xaxis=dict(type='category'))  # ensures discrete x-axis

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🕵🏻Insights:")
    st.markdown("""
    <p style='color: brown;'>
    The clustered bar chart from 2019 to 2023 highlights a relatively stable trend in both total area harvested and total production, with only a slight dip in 2023, suggesting consistent agricultural practices and land use. However, yield per hectare has remained low and flat, indicating stagnation in productivity despite high production volumes—pointing to reliance on land area rather than efficiency gains. This underscores a need for innovations such as improved seeds or farming techniques. Supporting this, crop-specific trends from 2019 to 2022 across various rankings reveal a mixed agricultural landscape. While many crops—like "Oil palm fruit," "Apples," and "Cassava"—exhibit steady or growing cultivated areas, others such as "Rye," "Safflower-seed oil," and "Poppy seed" show declines or flat trends. Additionally, several crops like "Ducks," "Beeswax," and "Blueberries" display notable volatility. These varied patterns reflect a dynamic and responsive agricultural sector shaped by shifting demand, environmental conditions, and policy influences. Furthermore, the wide disparity in cultivated land—ranging from thousands to millions of hectares—highlights the differing scales of crop importance and resource allocation. Overall, while agricultural land use is steady, enhancing yield efficiency remains a critical area for development.
    </p>
    """, unsafe_allow_html=True)
    
# --- Geographic Insights Page ---
elif page == "🌍 Geographic Insights":
    st.header("🌍 Geographic Insights by Crop")

    st.subheader("⛳High-Yielding Regions Based on Average Yield")
    query_map = """
        SELECT Area, AVG(Yield) AS avg_yield
        FROM crop_data
        GROUP BY Area
        ORDER BY avg_yield DESC;
        """
    df_map = execute_query(query_map)
    df_map_sorted = df_map.sort_values(by="avg_yield", ascending=False).reset_index(drop=True)
    midpoint = len(df_map_sorted) // 2
    top_50 = df_map_sorted.iloc[:midpoint]
    bottom_50 = df_map_sorted.iloc[midpoint:]
    view_option = st.radio("Select View:", ("Top 50% Yielding Areas", "Bottom 50% Yielding Areas"), horizontal=True)

# Filter data based on selection
    if view_option == "Top 50% Yielding Areas":
      data_to_plot = top_50
    else:
      data_to_plot = bottom_50
    fig = px.bar(
        data_to_plot,
        x='Area',
        y='avg_yield',
        color='avg_yield',
        color_continuous_scale='RdYlBu',
        title=f"{view_option}"
        )
    st.plotly_chart(fig)
    
#Area vs yeild
    st.subheader("🌱 Harvested Area vs Yield")
    grouped_df = (
       df
       .groupby(['Area', 'Item'], as_index=False)
       .agg(
          total_area=('Area harvested (ha)', 'sum'),
          avg_yield=('Yield (kg/ha)', 'mean')
        )
    )

# Filter out entries with no harvested area
    grouped_df = grouped_df[grouped_df['total_area'] > 0]

# Sort by average yield (ascending)
    grouped_df = grouped_df.sort_values('avg_yield', ascending=True)

# Radio button for filtering groups
    area_filter = st.radio(
       "Select Harvested Area Group",
       ("≤ 2.5M ha","2.5M - 5M ha", "5M - 10M ha", "10M - 25M ha", "25M - 50M ha", "> 50M ha"),
       horizontal=True
     )

# Apply filtering based on user selection
    if area_filter == "≤ 2.5M ha":
      filtered_df = grouped_df[grouped_df["total_area"] <= 2_500_000]
    elif area_filter == "2.5M - 5M ha":
      filtered_df = grouped_df[(grouped_df["total_area"] > 2_500_000) & (grouped_df["total_area"] <= 5_000_000)] 
    elif area_filter == "5M - 10M ha":
      filtered_df = grouped_df[(grouped_df["total_area"] > 5_000_000) & (grouped_df["total_area"] <= 10_000_000)] 
    elif area_filter == "10M - 25M ha":
      filtered_df = grouped_df[(grouped_df["total_area"] > 10_000_000) & (grouped_df["total_area"] <= 25_000_000)] 
    elif area_filter == "25M - 50M ha":
      filtered_df = grouped_df[(grouped_df["total_area"] > 25_000_000) & (grouped_df["total_area"] <= 50_000_000)] 
    else:
      filtered_df = grouped_df[grouped_df["total_area"] > 50_000_000]   

# Display in Streamlit
    

    fig = px.scatter(
      filtered_df,
      x="avg_yield",
      y="total_area",
      size="total_area",
      color="Area",
      hover_name="Item",
      title="Crop Area and Yield per Region (Sorted by Yield Ascending)",
      labels={
          "avg_yield": "Average Yield (kg/ha)",
          "total_area": "Harvested Area (ha)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)
 

    st.subheader("🌿Harvested Area vs Production")
#Area vs Production
    grouped_df = (
       df
       .groupby(['Area', 'Item'], as_index=False)
       .agg(
          total_area=('Area harvested (ha)', 'sum'),
          avg_production=('Production (t)', 'mean')
        )
    )

# Filter out entries with no harvested area
    grouped_df1 = grouped_df[grouped_df['total_area'] > 0]

# Sort by average yield (ascending)
    grouped_df1 = grouped_df1.sort_values('avg_production', ascending=True)

# Radio button for filtering groups
    area_filter1 = st.radio(
       "Select Harvested Area Group",
       ("≤ 2.5M ha","2.5M - 5M ha", "5M - 10M ha", "10M - 25M ha", "25M - 50M ha", "> 50M ha"),
       horizontal=True,
       key="area_filter_production"
     )

# Apply filtering based on user selection
    if area_filter1 == "≤ 2.5M ha":
      filtered_df1 = grouped_df[grouped_df["total_area"] <= 2_500_000]
    elif area_filter1 == "2.5M - 5M ha":
      filtered_df1 = grouped_df[(grouped_df["total_area"] > 2_500_000) & (grouped_df["total_area"] <= 5_000_000)] 
    elif area_filter1 == "5M - 10M ha":
      filtered_df1 = grouped_df[(grouped_df["total_area"] > 5_000_000) & (grouped_df["total_area"] <= 10_000_000)] 
    elif area_filter1 == "10M - 25M ha":
      filtered_df1 = grouped_df[(grouped_df["total_area"] > 10_000_000) & (grouped_df["total_area"] <= 25_000_000)] 
    elif area_filter1 == "25M - 50M ha":
      filtered_df1 = grouped_df[(grouped_df["total_area"] > 25_000_000) & (grouped_df["total_area"] <= 50_000_000)] 
    else:
      filtered_df1 = grouped_df[grouped_df["total_area"] > 50_000_000]   
 

 # Display in Streamlit
    fig = px.scatter(
      filtered_df1,
      x="avg_production",
      y="total_area",
      size="total_area",
      color="Area",
      hover_name="Item",
      title="Crop Area and Yield per Region",
      labels={
          "avg_production": "Average Production (t)",
          "total_area": "Harvested Area (ha)"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    # Get distinct crop items
    crop_query = "SELECT DISTINCT Item FROM crop_data ORDER BY Item"
    crop_list = execute_query(crop_query)["Item"].tolist()
    crop = st.selectbox("Select Crop", crop_list)

    # Production by Area
    prod_query = """
        SELECT Area, SUM(Production) AS Production
        FROM crop_data
        WHERE Item = %s
        GROUP BY Area
        ORDER BY Production DESC
    """
    prod_df = execute_query(prod_query, params=[crop])

    fig = px.bar(
        prod_df,
        x="Area",
        y="Production",
        title=f"Top Producing Areas for {crop}",
        labels={"Production": "Total Production (tons)"},
        color="Production",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Yield by Area
    yield_query = """
        SELECT Area, AVG(Yield) AS Yield
        FROM crop_data
        WHERE Item = %s
        GROUP BY Area
        ORDER BY Yield DESC
    """
    yield_df = execute_query(yield_query, params=[crop])

    fig = px.bar(
        yield_df,
        x="Area",
        y="Yield",
        title=f"Top Yielding Areas for {crop}",
        labels={"Yield": "Average Yield (kg/ha)"},
        color="Yield",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)
 
    st.subheader("🕵🏻Insights:")
    st.markdown("""
    <p style='color: brown;'>
    The visualizations comparing top and bottom 50% yielding areas, along with bubble plots of harvested area vs. yield and production, reveal critical insights into global agricultural efficiency. Countries with the largest cultivated areas—like India, China, and the United States—dominate in land use but often show relatively lower average yields, indicating a reliance on scale rather than efficiency. In contrast, nations with smaller land areas—especially those harvesting ≤2.5 million hectares—frequently achieve much higher yields, in some cases exceeding 200,000 kg/ha, likely due to intensive farming methods or high-value crops. The top-yielding countries, such as Iceland, Oman, and Norway, demonstrate that yield optimization through advanced technologies or efficient practices can far surpass productivity in larger nations. Additionally, while countries like Brazil and the U.S. combine moderate land use with high production, many others with vast agricultural areas underperform in yield, highlighting untapped potential. These charts collectively underscore that increasing productivity is less about expanding land and more about improving yield efficiency through innovation, crop selection, and sustainable practic
    </p>
    """, unsafe_allow_html=True)

elif page == "🧾Detailed Report":
    st.title("📈 Predicting Crop Production Based on Agricultural Data")
    st.markdown("An analytical overview of global crop performance, yield efficiency, and predictive modeling using Random Forest Regression.")

    # Section: Introduction
    st.header("🌍Introduction")
    st.markdown("""
    Agriculture remains a cornerstone of global food security, economic development, and rural livelihoods. However, disparities in yield, production efficiency, and land utilization persist between countries and crop types. This project, *Predicting Crop Production Based on Agricultural Data*, explores global patterns in crop performance, yield efficiency, and land use to identify actionable insights and enhance prediction models for agricultural productivity.
    """)

    # Section: Geographic Insights by Crop
    st.header("🗺️ Geographic Insights by Crop")

    st.subheader("🌾High-Yielding Regions Based on Average Yield")
    st.markdown("""
    The two bar charts illustrate a stark contrast in average yield between the bottom and top 50% yielding areas. Countries in the bottom half, such as Eritrea, Gambia, and Sudan, generally have yields below 7,000 kg/ha, often represented with lighter colors indicating lower productivity. These countries are predominantly developing nations, suggesting that limited access to modern agricultural technologies and infrastructure may be contributing to low yields.
    """)
    st.markdown("""
    In contrast, the top 50% includes countries like Iceland, Oman, and Norway, with Iceland standing out dramatically with an exceptionally high yield nearing 40,000 kg/ha. This highlights a significant disparity, with a few top performers skewing the distribution. The top-yielding nations are largely developed and may benefit from advanced technology, efficient practices, or favourable conditions.
    """)
    st.markdown("**🔍Observation:** The wide yield gap underscores the need for targeted support and knowledge transfer to lower-performing regions to enhance global agricultural productivity, equity, and for targeted investment.")

    st.subheader("🌾Harvested Area vs. Yield")
    st.markdown("""
    The bubble charts show that many countries with large harvested areas—such as India, the United States, Brazil, and China—tend to have moderate to low yields, clustering around the 5,000–10,000 kg/ha range. Interestingly, countries like Mexico and Kazakhstan demonstrate high yields (20,000–30,000 kg/ha) despite smaller harvested areas, suggesting effective agricultural strategies or specialized crops.
    """)
    st.markdown("""
    On the other end, countries with very small harvested areas (e.g., Eritrea, Sao Tome and Principe, and Brunei) show extreme yield values—either very low or exceptionally high—possibly due to niche crops or data anomalies.
    """)
    st.markdown("**🔍Observation:** The key takeaway is that large crop areas do not necessarily lead to high productivity, emphasizing the need to prioritize efficiency and technological adoption over scale alone.")

    st.subheader("🌾Harvested Area vs. Production")
    st.markdown("""
    Major producers such as the United States, China, and India dominate both in terms of harvested area and production, reflecting their extensive agricultural operations. Countries like Nigeria, Canada, and France show moderate production with relatively smaller harvested areas, suggesting more efficient land use. Meanwhile, smaller-scale producers in parts of Africa, Asia, and Latin America exhibit limited production and land utilization.
    """)
    st.markdown("**🔍Observation:**  This highlights a wide spectrum of agricultural performance, where high output doesn't always indicate high efficiency, reinforcing the importance of yield optimization strategies.")

    # Section: Annual Trends
    st.header("📊Annual Trends for All Crops (2019–2023)")
    st.markdown("""
    The clustered bar chart from 2019 to 2023 indicates stable trends in both total area harvested and total production, with a slight dip in 2023. This suggests relatively consistent agricultural practices and land use during this period. However, yield per hectare remains low and flat, pointing to stagnation in productivity. Despite high total outputs, efficiency has not improved significantly, suggesting a reliance on expanding land rather than innovating farming methods.
    """)
    st.markdown("""
    Supporting charts of individual crops from 2019 to 2022 reveal a mixed picture. While crops like Oil palm fruit, Apples, and Cassava exhibit growing cultivated areas, others like Rye, Safflower-seed oil, and Poppy seed show declines or plateaus. Some, such as Ducks,     Beeswax, and Blueberries, display notable volatility—indicating shifting demand or environmental sensitivity. These trends reflect a responsive agricultural sector shaped by market, policy, and climate conditions.
    """)
    st.markdown("**🔍Observation:** The variation in cultivated land from thousands to millions of hectares highlights disparities in crop scale and importance. While land use has remained stable, improving yield efficiency is the critical path forward for sustainable agriculture.")

    # Section: Crop-Level Performance Analysis
    st.header("🌱Crop-Level Performance Analysis")

    st.subheader("🔍Identifying Reliable High-Yield Crops Using Yield and Standard Deviation")
    st.markdown("""
    The bar chart compares crop yields alongside their standard deviations to assess both productivity and consistency. Crops like cucumbers, gherkins, and tomatoes lead with average yields above 20,000 kg/ha, but exhibit high variability (color-coded in red), indicating potential dependence on specific climates or regions.
    """)
    st.markdown("""
    In contrast, essential crops such as groundnuts, soybeans, and cotton deliver more stable but moderate yields (depicted in blue), offering reliability even if not peak productivity.
    """)
    st.markdown("**🔍Observation:** While high-yield crops can drive production gains, their unpredictability poses a risk. A balanced strategy blending stable and productive crops is crucial for sustainable agricultural planning.")

    st.subheader("🌾Crops Using Large Land Area but Yielding Less")
    st.markdown("""
    The scatter plot of harvested area versus average yield reveals a critical inefficiency: several crops consume vast land resources yet yield relatively little. These crops are positioned on the right side of the vertical dashed line (indicating large harvested area) and often fall below the 5,000 kg/ha yield threshold. They are marked in lighter blue, signaling low productivity per hectare.
    """)
    st.markdown("""
    On the other hand, higher-yielding crops tend to occupy significantly smaller areas, suggesting more efficient use of land.
    """)
    st.markdown("**🔍Observation:** This pattern highlights the need to reallocate land away from low-efficiency crops and towards those delivering higher returns per hectare, maximizing resource utilization in agriculture.")

    st.subheader("🌿Yield per Hectare by Crop Group")
    st.markdown("""
    The yield spectrum across various crop types reveals a distinct hierarchy in productivity. At the top end are high-yielding crops such as cucumbers, sugar cane, and tomatoes, which consistently produce over 20,000 kg/ha. Following these are fruits like cranberries and oranges, with solid performance around 8,000 kg/ha. A broad middle tier consists of vegetables, fruits, and oil crops yielding between 4,000 and 7,000 kg/ha, reflecting moderate productivity. Grains and pulses such as barley, oats, and groundnuts fall into the lower range, averaging close to 4,500 kg/ha. At the bottom of the spectrum are the least productive crops, including lentils, vanilla, and dried flowers, with yields typically under 4,200 kg/ha. A color gradient ranging from yellow to dark purple visually reinforces these yield differences, offering a clear and intuitive understanding of crop performance.
    """)
    st.markdown("**🔍Observation:** This classification provides a clear hierarchy of yield efficiency, enabling targeted decisions in crop selection, diversification, or substitution, depending on regional goals or constraints.")

    # Section: Predictive Modeling
    st.header("🤖Predictive Modeling: Random Forest Regression")
    st.markdown("""
    To better understand and forecast crop production, a Random Forest Regression model was developed using agricultural data. This ensemble method offers robustness and handles non-linear relationships effectively, making it well-suited for complex agricultural datasets.
    """)

    st.markdown("""
    **Model Performance:**
    - Train R² Score: 0.9917  
    - Test R² Score: 0.9738  
    - Mean Squared Error (MSE): *(insert value)*  
    - Mean Absolute Error (MAE): *(insert value)*  
    """)

    st.markdown("""
    **Interpretation:**  
    The model performs well and is suitable for forecasting production, identifying underperformers, and supporting planning.

    **Future Enhancements:**
    - Include weather/soil/fertilizer data
    - Try boosting models (e.g., XGBoost)
    - Add time-series patterns
    """)

    # Section: Conclusion
    st.header("Conclusion")
    st.markdown("""
    This report has combined visual analytics and machine learning to derive insights into global crop production. The geographic and crop-level analysis revealed disparities in yield and land efficiency, while the regression model showed strong predictive potential.
    """)
    st.markdown("""
    Moving forward, boosting yield efficiency through innovation, data-driven policy, and technology adoption is critical to achieving global food security and sustainable agriculture.
    """)





   





   

  




