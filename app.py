import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("📊 Optimized Glass (Verre) Material Experiments")

# Load datasets
df_existing = pd.read_csv("cleaned_verredata.csv")
df_generated = pd.read_csv("generated_verre_experiments.csv")

# Define approximate material costs per metric ton (in USD)
material_costs_per_ton = {
    "Ciment": 125,  # $100-$150 per ton
    "Argile": 40,   # $30-$50 per ton
    "Alumine": 350, # $300-$400 per ton
    "Verre": 35,    # $20-$50 per ton
    "Chamotte": 300 # $200-$400 per ton
}

# Assume **one piece weighs 2 kg** (adjustable)
piece_weight_kg = 2  # Change if needed
kg_per_ton = 1000  # 1 ton = 1000 kg

# Function to calculate total cost per piece
def calculate_cost_per_piece(row):
    cost_per_ton = sum(row[material] * material_costs_per_ton[material] / 100 for material in material_costs_per_ton.keys())
    cost_per_kg = cost_per_ton / kg_per_ton  # Convert to cost per kg
    return cost_per_kg * piece_weight_kg  # Convert to cost per piece

# Show best existing experiment (lowest conductivity)
st.write("### ✅ Best Existing Experiment (Lowest Conductivity)")
best_existing = df_existing.loc[df_existing["Conductivité thermique"].idxmin()]
best_existing_cost_per_piece = calculate_cost_per_piece(best_existing)
st.dataframe(best_existing.to_frame().T)
st.write(f"💰 **Estimated Cost per Piece:** ${best_existing_cost_per_piece:.2f}")

# Show best generated experiment (lowest conductivity)
st.write("### 🔬 Best Newly Generated Experiment")
best_generated = df_generated.loc[df_generated["Conductivité thermique"].idxmin()]
best_generated_cost_per_piece = calculate_cost_per_piece(best_generated)
st.dataframe(best_generated.to_frame().T)
st.write(f"💰 **Estimated Cost per Piece:** ${best_generated_cost_per_piece:.2f}")

# Show all new experiments sorted
st.write("### 📈 All Newly Generated Experiments (Sorted)")
st.dataframe(df_generated.sort_values(by=["Conductivité thermique", "Résistance Mécanique"], ascending=[True, False]))

# ---- 📥 DOWNLOAD BUTTONS ---- #
st.write("## 📥 Download Experiment Data")
col1, col2 = st.columns(2)

# Download button for existing experiments
with col1:
    st.download_button(
        label="📥 Download Existing Experiments",
        data=df_existing.to_csv(index=False).encode("utf-8"),
        file_name="existing_verre_experiments.csv",
        mime="text/csv"
    )

# Download button for newly generated experiments
with col2:
    st.download_button(
        label="📥 Download Generated Experiments",
        data=df_generated.to_csv(index=False).encode("utf-8"),
        file_name="generated_verre_experiments.csv",
        mime="text/csv"
    )

# ---- 📊 VISUALIZATIONS ---- #
st.write("## 🔍 Data Distribution & Analysis")

# 🔹 Histogram: Distribution of Conductivity & Resistance
st.write("### 📏 Distribution of Conductivity & Resistance")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df_existing["Conductivité thermique"], bins=20, kde=True, ax=ax[0], color="blue")
ax[0].set_title("Existing: Conductivity Distribution")
sns.histplot(df_existing["Résistance Mécanique"], bins=20, kde=True, ax=ax[1], color="red")
ax[1].set_title("Existing: Resistance Distribution")
st.pyplot(fig)

# 🔹 Scatter Plot: Conductivity vs. Resistance
st.write("### 🔬 Relationship Between Conductivity & Resistance")
fig = px.scatter(
    df_existing, x="Conductivité thermique", y="Résistance Mécanique",
    color="Conductivité thermique", size="Résistance Mécanique",
    title="Conductivity vs. Resistance (Existing Data)"
)
st.plotly_chart(fig)

# 🔹 Box Plot: Compare Composition Ranges
st.write("### 📦 Box Plot of Compositions in Experiments")
fig, ax = plt.subplots(figsize=(12, 6))
df_melted = df_existing.melt(id_vars=["Conductivité thermique", "Résistance Mécanique"], var_name="Component", value_name="Percentage")
sns.boxplot(x="Component", y="Percentage", data=df_melted, ax=ax)
ax.set_title("Composition Variability Across Experiments")
st.pyplot(fig)

# 🔹 Correlation Heatmap
st.write("### 🔥 Heatmap: Correlation Between Factors & Results")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df_existing.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)

st.write("## ✅ Insights & Recommendations")
st.markdown(
    """
    - **Low Conductivity & High Resistance Trends**: Higher **Alumine & Verre** tend to result in **lower conductivity**.
    - **Trade-offs**: Increasing certain components (e.g., **Ciment, Argile**) may improve mechanical resistance but could negatively impact thermal conductivity.
    - **Optimization Needed**: Fine-tuning the component ratios can further lower conductivity while maintaining resistance.
    - **Cost Considerations**: Lowering **Alumine & Chamotte** may reduce costs, but at the expense of material performance.
    """
)
