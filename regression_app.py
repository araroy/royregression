import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Title
st.title("Advanced Regression Tool")
st.markdown("""
This tool offers:
1. File upload and dynamic variable creation.
2. Data cleaning (merge, mean, sum, etc.).
3. Descriptive statistics with visualizations.
4. Regression analysis (Multiple Regression and Logistic Regression).
5. Trendline plots and downloadable results.
""")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Load the dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    
    # Initialize session state
    if "df" not in st.session_state:
        st.session_state["df"] = df
    else:
        df = st.session_state["df"]

    st.write("Preview of Uploaded Data:")
    st.write(st.session_state["df"].head())

    # Data Cleaning Section
    st.markdown("### Data Cleaning")
    operation = st.selectbox("Choose a cleaning operation", ["None", "Merge Columns (Remove Blanks)", "Mean of Columns", "Sum of Columns"])

    if operation == "Merge Columns (Remove Blanks)":
        col1 = st.selectbox("Select First Column", options=st.session_state["df"].columns)
        col2 = st.selectbox("Select Second Column", options=st.session_state["df"].columns)
        new_var = st.text_input("New Variable Name", "merged_variable")
        if st.button("Merge Columns"):
            st.session_state["df"][new_var] = st.session_state["df"][col1].combine_first(st.session_state["df"][col2])
            st.success(f"New variable '{new_var}' created!")

    elif operation == "Mean of Columns":
        cols = st.multiselect("Select Columns", options=st.session_state["df"].columns)
        new_var = st.text_input("New Variable Name", "mean_variable")
        if st.button("Create Mean Variable"):
            st.session_state["df"][new_var] = st.session_state["df"][cols].mean(axis=1)
            st.success(f"New variable '{new_var}' created!")

    elif operation == "Sum of Columns":
        cols = st.multiselect("Select Columns", options=st.session_state["df"].columns)
        new_var = st.text_input("New Variable Name", "sum_variable")
        if st.button("Create Sum Variable"):
            st.session_state["df"][new_var] = st.session_state["df"][cols].sum(axis=1)
            st.success(f"New variable '{new_var}' created!")

    st.write("Updated Data:")
    st.write(st.session_state["df"].head())

    # Descriptive Statistics Section
    st.markdown("### Descriptive Statistics")
    var_to_describe = st.selectbox("Select Variable for Descriptives", options=st.session_state["df"].columns)
    if st.button("Show Descriptives"):
        if pd.api.types.is_numeric_dtype(st.session_state["df"][var_to_describe]):
            mean_val = st.session_state["df"][var_to_describe].mean()
            std_val = st.session_state["df"][var_to_describe].std()
            st.write(f"**Mean**: {mean_val:.2f}, **Standard Deviation**: {std_val:.2f}")
            fig, ax = plt.subplots()
            sns.histplot(st.session_state["df"][var_to_describe], kde=True, ax=ax)
            ax.set_title(f"Distribution of {var_to_describe}")
            st.pyplot(fig)
        else:
            freq_table = st.session_state["df"][var_to_describe].value_counts()
            st.write("Frequency Table:")
            st.write(freq_table)
            fig, ax = plt.subplots()
            freq_table.plot(kind="bar", ax=ax, color="skyblue")
            ax.set_title(f"Frequency Chart of {var_to_describe}")
            st.pyplot(fig)

    # Graphical Visualization Section
    st.markdown("### Graphical Visualization with Trendline")
    dv = st.selectbox("Select Dependent Variable (Y-axis)", options=st.session_state["df"].columns)
    iv = st.selectbox("Select Independent Variable (X-axis)", options=[col for col in st.session_state["df"].columns if col != dv])
    if st.button("Show Trendline"):
        if pd.api.types.is_numeric_dtype(st.session_state["df"][dv]) and pd.api.types.is_numeric_dtype(st.session_state["df"][iv]):
            fig, ax = plt.subplots()
            sns.regplot(x=st.session_state["df"][iv], y=st.session_state["df"][dv], ax=ax, ci=None, scatter_kws={"color": "skyblue"}, line_kws={"color": "red"})
            ax.set_title(f"Trendline: {dv} vs {iv}")
            ax.set_xlabel(iv)
            ax.set_ylabel(dv)
            st.pyplot(fig)
        else:
            st.error("Both the dependent and independent variables must be numeric.")

    # Regression Analysis Section
    st.markdown("### Regression Analysis")
    analysis_type = st.radio("Choose Analysis Type", ["Multiple Regression", "Logistic Regression"])
    predictors = st.multiselect("Select Independent Variables", options=[col for col in st.session_state["df"].columns if col != dv])
    interaction_term = st.checkbox("Include Interaction Term")

    if interaction_term and len(predictors) >= 2:
        interaction_vars = st.multiselect("Select Two Variables for Interaction", options=predictors, max_selections=2)
        if len(interaction_vars) == 2:
            interaction_name = f"{interaction_vars[0]}_x_{interaction_vars[1]}"
            st.session_state["df"][interaction_name] = st.session_state["df"][interaction_vars[0]] * st.session_state["df"][interaction_vars[1]]
            predictors.append(interaction_name)

    if st.button("Run Regression"):
        try:
            if analysis_type == "Multiple Regression":
                X = sm.add_constant(st.session_state["df"][predictors])
                y = st.session_state["df"][dv]
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
                X = sm.add_constant(st.session_state["df"][predictors])
                y = st.session_state["df"][dv]
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
