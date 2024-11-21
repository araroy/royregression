import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Title
st.title("Advanced Statistical Analysis Tool")
st.markdown("""
This tool offers:
1. Data Cleaning: Create new variables by merging, summing, or averaging.
2. Descriptive Statistics: Compute and visualize mean, standard deviation, and frequency charts.
3. Regression Analysis: Perform Multiple Regression and Logistic Regression with interactions.
4. Results Visualization: Scatter plots with trendlines and downloadable result tables.
""")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Load the dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    st.write("Preview of Uploaded Data:")
    st.write(df.head())

    # Data Cleaning Section
    st.markdown("### Data Cleaning")
    operation = st.selectbox("Choose a cleaning operation", ["None", "Merge Columns (Remove Blanks)", "Mean of Columns", "Sum of Columns"])

    if operation == "Merge Columns (Remove Blanks)":
        col1 = st.selectbox("Select First Column", options=df.columns)
        col2 = st.selectbox("Select Second Column", options=df.columns)
        new_var = st.text_input("New Variable Name", "merged_variable")
        if st.button("Merge Columns"):
            df[new_var] = df[col1].combine_first(df[col2])
            st.success(f"New variable '{new_var}' created!")

    elif operation == "Mean of Columns":
        cols = st.multiselect("Select Columns", options=df.columns)
        new_var = st.text_input("New Variable Name", "mean_variable")
        if st.button("Create Mean Variable"):
            df[new_var] = df[cols].mean(axis=1)
            st.success(f"New variable '{new_var}' created!")

    elif operation == "Sum of Columns":
        cols = st.multiselect("Select Columns", options=df.columns)
        new_var = st.text_input("New Variable Name", "sum_variable")
        if st.button("Create Sum Variable"):
            df[new_var] = df[cols].sum(axis=1)
            st.success(f"New variable '{new_var}' created!")

    st.write("Updated Data:")
    st.write(df.head())

    # Descriptive Statistics
    st.markdown("### Descriptive Statistics")
    var_to_describe = st.selectbox("Select Variable for Descriptives", options=df.columns)
    if st.button("Show Descriptives"):
        if pd.api.types.is_numeric_dtype(df[var_to_describe]):
            mean_val = df[var_to_describe].mean()
            std_val = df[var_to_describe].std()
            st.write(f"**Mean**: {mean_val:.2f}, **Standard Deviation**: {std_val:.2f}")
            fig, ax = plt.subplots()
            sns.histplot(df[var_to_describe], kde=True, ax=ax)
            ax.set_title(f"Distribution of {var_to_describe}")
            st.pyplot(fig)
        else:
            freq_table = df[var_to_describe].value_counts()
            st.write("Frequency Table:")
            st.write(freq_table)
            fig, ax = plt.subplots()
            freq_table.plot(kind="bar", ax=ax, color="skyblue")
            ax.set_title(f"Frequency Chart of {var_to_describe}")
            st.pyplot(fig)

    # Graphical Visualization: Trendline
    st.markdown("### Graphical Visualization with Trendline")
    dv = st.selectbox("Select Dependent Variable (Y-axis)", options=df.columns)
    iv = st.selectbox("Select Independent Variable (X-axis)", options=[col for col in df.columns if col != dv])
    if st.button("Show Trendline"):
        if pd.api.types.is_numeric_dtype(df[dv]) and pd.api.types.is_numeric_dtype(df[iv]):
            fig, ax = plt.subplots()
            sns.regplot(x=df[iv], y=df[dv], ax=ax, ci=None, scatter_kws={"color": "skyblue"}, line_kws={"color": "red"})
            ax.set_title(f"Trendline: {dv} vs {iv}")
            ax.set_xlabel(iv)
            ax.set_ylabel(dv)
            st.pyplot(fig)
        else:
            st.error("Both the dependent and independent variables must be numeric.")

    # Regression Analysis
    st.markdown("### Regression Analysis")
    analysis_type = st.radio("Choose Analysis Type", ["Multiple Regression", "Logistic Regression"])
    predictors = st.multiselect("Select Independent Variables", options=[col for col in df.columns if col != dv])
    interaction_term = st.checkbox("Include Interaction Term")

    if interaction_term and len(predictors) >= 2:
        interaction_vars = st.multiselect("Select Two Variables for Interaction", options=predictors, max_selections=2)
        if len(interaction_vars) == 2:
            interaction_name = f"{interaction_vars[0]}_x_{interaction_vars[1]}"
            df[interaction_name] = df[interaction_vars[0]] * df[interaction_vars[1]]
            predictors.append(interaction_name)

    if st.button("Run Regression"):
        try:
            if analysis_type == "Multiple Regression":
                # Fit multiple regression model
                X = sm.add_constant(df[predictors])
                y = df[dv]
                model = sm.OLS(y, X).fit()
                st.markdown("### Regression Results")
                st.write(model.summary())

                # Download results
                results_df = pd.DataFrame({
                    "Predictor": model.params.index,
                    "Coefficient": model.params.values,
                    "Std. Error": model.bse.values,
                    "p-value": model.pvalues.values
                })
                results_df["R-squared"] = model.rsquared
                st.write(results_df)

                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Regression Results as CSV", data=csv, file_name="regression_results.csv", mime="text/csv")

            elif analysis_type == "Logistic Regression":
                # Fit logistic regression model
                X = sm.add_constant(df[predictors])
                y = df[dv]
                model = sm.Logit(y, X).fit()
                st.markdown("### Logistic Regression Results")
                st.write(model.summary())

                # Download results
                results_df = pd.DataFrame({
                    "Predictor": model.params.index,
                    "Coefficient": model.params.values,
                    "Std. Error": model.bse.values,
                    "p-value": model.pvalues.values
                })
                results_df["Pseudo R-squared"] = model.prsquared
                st.write(results_df)

                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Logistic Regression Results as CSV", data=csv, file_name="logistic_regression_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error in regression analysis: {e}")
