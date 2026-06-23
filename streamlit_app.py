import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

API_URL = os.environ.get('FASTAPI_URL', 'http://127.0.0.1:8000')

st.set_page_config(page_title='Titanic Survival Predictor', layout='wide')

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #5BA4CF;
}
}
}
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_train_data():
    return pd.read_csv('data/train.csv')


def render_home():
    st.title('Titanic Survival Predictor')

    col1, col2, col3 = st.columns(3)
    col1.metric('Raw rows', '891')
    col2.metric('Raw features', '11')
    col3.metric('Engineered features', '20')

    st.subheader('Champion model — LogReg_Baseline')
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'Macro F1', 'ROC-AUC'],
        'Score': [0.8436, 0.8254, 0.7536, 0.8320, 0.8725],
    })
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    st.info(
        'Logistic Regression beat both Random Forest and tuned Gradient '
        'Boosting on this split.'
    )


def render_predict():
    st.title('Predict')

    with st.form('predict_form'):
        c1, c2 = st.columns(2)
        with c1:
            pclass = st.selectbox('Passenger class', [1, 2, 3], index=2)
            sex = st.radio('Sex', ['male', 'female'])
            age = st.slider('Age', 0, 100, 30)
            embarked = st.selectbox('Port of embarkation', ['S', 'C', 'Q'])
        with c2:
            sibsp = st.number_input('Siblings / spouses aboard', min_value=0, value=0)
            parch = st.number_input('Parents / children aboard', min_value=0, value=0)
            fare = st.number_input('Fare paid', min_value=0.0, value=32.0, step=1.0)

        submitted = st.form_submit_button('Predict survival')

    if submitted:
        payload = {
            'pclass': pclass, 'sex': sex, 'age': age,
            'sibsp': sibsp, 'parch': parch, 'fare': fare, 'embarked': embarked,
        }
        try:
            with st.spinner('Calling the API...'):
                r = requests.post(f'{API_URL}/predict', json=payload, timeout=5)
            if r.status_code == 200:
                result = r.json()
                prob = result['probability']
                if result['survived'] == 1:
                    st.success(result['verdict'])
                else:
                    st.error(result['verdict'])
                st.progress(prob, text=f'Survival probability: {prob:.1%}')
            else:
                st.error(f'API returned status {r.status_code}: {r.text}')
        except requests.exceptions.ConnectionError:
            st.error(
                f"Couldn't reach the API at {API_URL}. "
                'Make sure the FastAPI server is running.'
            )
        except requests.exceptions.Timeout:
            st.error('The API took too long to respond. Try again.')


def render_eda():
    st.title('Exploratory Data Analysis')

    eda_path = 'reports/eda_overview.png'
    if os.path.exists(eda_path):
        st.image(eda_path, caption='Step 2 — 6-chart EDA overview', use_column_width=True)
    else:
        st.warning(f'{eda_path} not found — run the notebook through Step 2 to generate it.')

    st.subheader('Interactive view — survival rate by feature')
    df = load_train_data()
    feature = st.selectbox('Break down survival by', ['Pclass', 'Sex', 'Embarked'])
    rate_df = df.groupby(feature, dropna=False)['Survived'].mean().reset_index()
    rate_df['Survived'] = (rate_df['Survived'] * 100).round(1)
    fig = px.bar(
        rate_df, x=feature, y='Survived',
        labels={'Survived': 'Survival rate (%)'},
        title=f'Survival rate by {feature}',
        color='Survived', color_continuous_scale='Blues',
    )
    st.plotly_chart(fig, use_container_width=True)


def render_model_insights():
    st.title('Model Insights — SHAP')
    st.caption('Feature impact for the champion model, computed on the validation set')

    col1, col2 = st.columns(2)
    beeswarm_path = 'reports/shap_beeswarm.png'
    bar_path = 'reports/shap_bar.png'

    with col1:
        if os.path.exists(beeswarm_path):
            st.image(beeswarm_path, caption='SHAP summary (beeswarm)', use_column_width=True)
        else:
            st.warning(f'{beeswarm_path} not found.')

    with col2:
        if os.path.exists(bar_path):
            st.image(bar_path, caption='Mean |SHAP value| — top 15 features', use_column_width=True)
        else:
            st.warning(f'{bar_path} not found.')

    st.subheader('Key Takeaways:')
    st.markdown(
        '- **Title (especially "Mr")** is the single strongest driver of predicted '
        'survival, followed closely by **Fare** and **Pclass_3**.\n'
        '- **Age** and **Deck** also contribute meaningfully, while raw **Sex** '
        'splits its influence between the `Sex_male` and `Sex_female` one-hot columns.\n'
        '- This reflects the historical "women and children first, higher class '
        'first" evacuation pattern. Title acts as a compact proxy for several '
        'of these factors at once.'
    )


PAGES = {
    'Home': render_home,
    'Predict': render_predict,
    'EDA': render_eda,
    'Model Insights': render_model_insights,
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio('Go to', list(PAGES.keys()))
PAGES[selection]()
