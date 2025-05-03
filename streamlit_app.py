import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import dagshub
import mlflow
from streamlit_option_menu import option_menu
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, precision_score
import shap
from mlflow import log_metric, log_param, log_artifact
from sklearn.model_selection import cross_val_score


st.title("Preventing Obesity 101: What To Look For!")
st.write("How can we reduce obesity? What are the largest factors that contribute to obesity and what should scientific and medical professionals target?")
st.image("veggies.webp")


selected = option_menu(menu_title=None, options=["01 Introduction", "02 Data Visualization", "03 Predictions", "04 Conclusion"], default_index=0, orientation="horizontal")
df = pd.read_csv("obesity.csv")


st.title("Let's Get Down To Business")


if selected == "01 Introduction":
    st.markdown("""
    **Introduction:**
    Obesity, defined by the World Health Organization as an excessive accumulation of fat that presents a risk to health, is a common health issue across the world. It is often caused by a variety of factors including daily lifestyle habits, genetic history, and physical makeup. Moreover, obesity is a well-known risk factor for diabetes, cancer, and other serious health issues, so reducing its prevalence around the world proves to be a critical task to help improve health. Our analysis aims to pinpoint the habits and lifestyle factors most strongly correlated with the outcome of obesity, our models aim to predict whether someone will be obese based on these factors. We aim to use this data to shape public health preventive measures, such as anti-smoking campaigns and healthy food advertisements. How can we reduce obesity? What are the largest factors contributing to obesity, and what variables should scientific and medical professionals target?
    """)

    dataset_variables = {
    "Gender": "Male or Female.",
    "Age": "The person’s age in years.",
    "Height": "Height in meters.",
    "Weight": "Weight in kilograms.",
    "family_history_with_overweight": "Whether the person has a family history...",
    "FAVC": "If the person frequently consumes high-calorie foods (yes/no).",
    "FCVC": "Frequency of vegetable consumption...",
    "NCP": "Number of main meals per day.",
    "CAEC": "Frequency of consuming food between meals...",
    "SMOKE": "Whether the person smokes (yes/no).",
    "CH2O": "Daily water intake...",
    "SCC": "If the person monitors their calorie intake...",
    "FAF": "Physical activity frequency...",
    "TUE": "Time spent using technology...",
    "CALC": "Frequency of alcohol consumption...",
    "MTRANS": "Main mode of transportation...",
    "NObeyesdad": "Obesity level..."
    }
    st.markdown("### :grey[Dataset Variables]")
    st.table(pd.DataFrame.from_dict(dataset_variables, orient='index', columns=["Description"]))
    st.markdown("### :grey[Data Exploration]")
    num = st.number_input("No of rows", 5, 10)
    st.dataframe(df.head(20))
    st.dataframe(df.describe())
    st.write(df.shape)
    # Pairplot
    st.markdown("## Pairplot")
    st.image("pairplot.png")



# Page 2
if selected == "02 Data Visualization":
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Correlation", "Violin Plot", "Scatter Plot", "Bar Chart", "Bar Chart"])

    with tab1:
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        selected_vars = st.multiselect("Select variables for the correlation matrix", numeric_columns, default=numeric_columns[:3])
        st.markdown("### :green[Correlation Matrix]")
        if len(selected_vars) > 1:
            correlation = df[selected_vars].corr()
            fig = px.imshow(correlation, text_auto=True, color_continuous_scale="Greens")
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        else:
            st.warning("Please select at least two variables.")
    with tab2:
        st.markdown("### :green[Violin Plot of Obesity Level by Frequency of Alcohol Consumption]")
        fig, ax = plt.subplots()
        sns.violinplot(x='CALC', y='NObeyesdad', data=df, ax=ax, palette='Greens')
        ax.set_xlabel('Frequency of Alcohol Consumption')
        ax.set_ylabel('Obesity Level')
        st.pyplot(fig)
        st.write("The only person in our dataset who always consumes alcohol is “normal weight” which is contradictory to our expectations because we thought that alcohol consumption contributes to being overweight. This graph proves to us that alcohol is not the only, maybe not even the main contributor to being overweight, as there are some people in our dataset who “frequently” consume alcohol and are classified only as Overweight_Level_I, while some who drink “sometimes” are classified into Overweight_Level_II. Very few who do not drink are in the Obesity_Level_III category, meaning that alcohol consumption is not as strongly correlated with obesity as other variables.")
    with tab3:
        st.markdown("### :green[Scatter Plot of Frequency of Physical Activity vs. Weight]")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="FAF", y="Weight", color="green", alpha=0.3, ax=ax)
        ax.set_xlabel('Physical Activity Frequency (0-3)')
        ax.set_ylabel('Weight (Kg)')
        st.pyplot(fig)
        st.write("While the scatterplot shows no strong correlation between increased physical activity and lower body weight, this doesn't necessarily mean that physical activity is not effective or not correlated with obesity. The physical activity variable only reflects frequency, not duration, intensity, or type, making it difficult to assess the true impact of someone’s physical activity on their weight. Training goals and routines are not captured by frequency, and weight does not capture the type of weight. For example, cardio exercises are typically associated with weight loss, while weight training can increase weight due to muscle gain, even if fat mass decreases. Since muscle weighs more than fat, physically active individuals may have higher weight but lower body fat percentage–something not captured by weight alone. ")
    with tab4:
        st.markdown("### :green[Bar Chart of Average Number of Main Meals Daily Between Those With and Without a Family History of Being Overweight]")
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="family_history_with_overweight", y="NCP", color="green", alpha=0.3, ax=ax, estimator=np.mean)
        ax.set_xlabel('Family History of Overweight?')
        ax.set_ylabel('Number of Main Meals per Day')
        st.pyplot(fig)
        st.write("On average, individuals with a family history of obesity consume approximately 2.7 meals per day, while those without such a history consume closer to 2.4 meals per day. This suggests that there is a positive correlation between family history of obesity and meal frequency, and more broadly reflects how individuals may have a genetic predisposition for greater food consumption. However, this graph lacks the nuance necessary to examine cultural influences or learned habits that may emerge from the environment, not genetics.")
    with tab5:
        st.markdown("### :green[Bar Chart of Average Number of Main Meals Daily by Obesity Level]")
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="NCP", y="NObeyesdad", color="green", alpha=0.3, ax=ax, estimator=np.mean)
        ax.set_xlabel('Number of Main Meals per Day')
        ax.set_ylabel('Obesity Level')
        st.pyplot(fig)
        st.write("Individuals who consume, on average, three meals per day are most likely to be classified as Obesity_Type_III, the highest obesity category. However, those classified as Insufficient_Weight also report consuming around 2.8 meals per day, and those classified as Normal_Weight consume around 2.6 meals per day. This suggests that meal frequency is not the most indicative variable of obesity, as it does not encompass portion size, calorie density, metabolism, and food choices. Moreover, for those who are of a lower socioeconomic status, there are many barriers to proper food consumption. For example, food stamps may not cover the cost of healthier food, or people may live in food deserts, where they are unable to access a grocery store.")


categorical_cols = df.select_dtypes(include=["object"]).columns.drop("NObeyesdad")
label_encoder = LabelEncoder()
for col in categorical_cols:
   df[col] = label_encoder.fit_transform(df[col])


# Encode NObeyesdad separately
nobesity_encoder = LabelEncoder()
df["NObeyesdad"] = nobesity_encoder.fit_transform(df["NObeyesdad"])

label_encoder = LabelEncoder()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])
    categorical_cols = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"]




if selected == "03 Predictions":
    tab6, tab7, tab8, tab9 = st.tabs(["Linear Regression", "Decision Tree", "Feature Importance/AI Explainability", "Hyperparameter Tuning"])

    with tab6:
        df = df.dropna()

        # Features and Target
        X = df.drop("Weight", axis=1)
        y = df["Weight"]

        st.header(":green[Linear Regression Classifier 📈]")
        # Train/Test Split  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        st.pyplot(fig)
        
        st.markdown(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")
        st.markdown(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")

    with tab7:
        st.header(":green[Decision Tree Classifier]")

        # Choose max depth for user adjustment
        max_depth_user = st.slider("Select max_depth:", min_value=1, max_value=20, value=3)

        # Target = NObeyesdad
        X = df.drop("NObeyesdad", axis=1)
        y = df["NObeyesdad"]

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Model training with user selected max_depth
        model = DecisionTreeClassifier(max_depth=max_depth_user, random_state=1)
        model.fit(X_train, y_train)

        # Predict and decode
        y_pred = model.predict(X_test)
        y_test_original = nobesity_encoder.inverse_transform(y_test)
        y_pred_original = nobesity_encoder.inverse_transform(y_pred)

        # Metrics
        st.success(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        st.success(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")

        # Cross-validated accuracy
        st.markdown("## Cross-Validated Accuracy")
        mean_cv_accuracy = np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy'))
        st.success(f"Mean CV Accuracy: {mean_cv_accuracy:.4f}")

        # Show tree for user-adjusted max_depth
        st.markdown("## Decision Tree Diagram")
        dot_data = export_graphviz(
            model,
            out_file=None,
            feature_names=X.columns,
            class_names=[str(nobesity_encoder.inverse_transform([i])[0]) for i in model.classes_],
            filled=True,
            rounded=True,
            special_characters=True
        )
        graph = graphviz.Source(dot_data)
        st.graphviz_chart(graph)

        # Button to run MLflow and log the results
        if st.button("Run MLflow"):
            dagshub.init(repo_owner='jmh10120', repo_name='dsfinal', mlflow=True)
            param_grid = {"max_depth": list(range(1, 21))}

            grid_search = GridSearchCV(
                DecisionTreeClassifier(random_state=1),
                param_grid,
                cv=5,
                scoring='accuracy'
            )

            mlflow.start_run()  # Start MLflow run

            # Run grid search
            grid_search.fit(X_train, y_train)

            # Get best model and results
            best_model = grid_search.best_estimator_
            best_depth = grid_search.best_params_["max_depth"]
            best_score = grid_search.best_score_

            # Show best results in Streamlit
            st.success(f"✅ Best Max Depth: {best_depth}")
            st.success(f"✅ Best Cross-Validated Accuracy: {best_score:.4f}")

            # Log each model's result to MLflow
            for max_depth in param_grid["max_depth"]:
                with mlflow.start_run(nested=True, run_name=f"Decision Tree {max_depth}"):
                    model = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mlflow.log_param("model_name", "DecisionTree")
                    mlflow.log_param("max_depth", max_depth)
                    mlflow.log_params(model.get_params())
                    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
                    mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
                    mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
                    mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, y_pred)))
                mlflow.end_run()  # End each nested run

        # Link to logged results in Dagshub (MLflow)
        st.markdown("----")
        st.markdown("## 📊 View Logged Results")
        st.markdown("[**View All Runs in MLflow (Dagshub)**](https://dagshub.com/jmh10120/dsfinal.mlflow)")

    with tab8:
        st.markdown("## :green[SHAP Feature Importance]")
        st.image("shap.jpeg")
    with tab9:
        st.markdown("## :green[MLFLOW Model Accuracy]")
        st.image("mlflowpic.png")


if selected == "04 Conclusion":
    st.title("Conclusion")
    st.markdown("""
    **Data Quality:**
    - While the dataset was large, its quality was questionable. For example, only one person answered “always” to drinking alcohol–this indicates that there may be an issue or bias resulting from data collection–it was not indicated whether the data collected was anonymous and individuals may have felt inclined to lie due to social pressures. This also suggests that the way these questions may have been asked skewed the data. 
    - Socioeconomic status (SES) was not included in our dataset. However, socioeconomic status tends to be a powerful indicator of obesity levels. For example, individuals with lower SES often live in neighborhoods that do not offer proper access to gyms or healthy grocery stores. Moreover, if they are living on food stamps, they may not have enough to afford other types of food.  The best deals are always on fast food, so families might feel more inclined to buy that rather than a healthy meal, especially if they cannot afford it. McDonald’s Happy Meals, for example, are around $4. If people simply need calories to survive, choosing McDonald’s may make more sense to these individuals. 
    """)
    st.markdown("""
    **Key Insights:**
    - Feature Importance (SHAP): In the SHAP feature importance graph, we see that weight has the most value in predicting obesity level, which means it is the most important variable. This is why we used it in our linear regression model. The next two most valuable variables would be height and gender followed by age. All the other variables barely have any effect on the output of obesity. We think that if socioeconomic status was included in this dataset, we would see a much clearer picture and could improve the accuracy of our model, as it would contain a variable that is extremely relevant and historically directly correlated with obesity rates. 
    - Linear Regression Classifier: As we can see by the linear regression classifier, the line of best fit is very close to the data being outputted. This means that the actual versus the predicted number of weight were relatively close together. Our R^2 coefficient is .56, which could be improved since there are still a lot of points far away from the line, however, our data is generally moving in the same direction as the line of best fit, which is a good start. In terms of real-world application, this means that our model is not very accurate when predicting weight, meaning it is not entirely reliable when determining how overweight someone is, however, it gives us a general idea of which range someone falls in, so this can still be helpful in determining whether or not someone may need an intervention. 
    - Decision Tree Diagram: Our ideal decision tree predictor contains 3 branches, and some of the gini coefficients on the lowest branch were close to 0.2, which is near perfect. This is what helps the classifier determine what the ideal depth is, because after the branches go back to containing higher gini coefficients, the tree is no longer optimal length. 
    """)
    st.image("1wrf0n.jpg")