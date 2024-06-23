# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# 폰트 설정
import matplotlib as mpl
mpl.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# Streamlit 애플리케이션 설정
st.title("투수 통계 회귀 분석")

# 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 선택하세요", type=["csv"])

if uploaded_file is not None:
    # 데이터프레임 생성
    picher = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기:")
    st.write(picher.head())

    with st.expander("데이터프레임 특성 요약"):
        # 데이터프레임 요약
        st.write(picher.describe())

        # 연봉 단위 확인 및 정규화
        st.write("연봉 데이터 정규화:")
        picher['연봉(2017)'] = picher['연봉(2017)'] * 1000
        picher['연봉(2018)'] = picher['연봉(2018)'] * 1000
        st.write(picher[['연봉(2017)', '연봉(2018)']].describe())

    with st.expander("데이터 전처리 및 피처 스케일링"):
        # 데이터 전처리 및 특성 스케일링 (연봉(2017) 제외)
        def standard_scaling(df, scale_columns):
            for col in scale_columns:
                series_mean = df[col].mean()
                series_std = df[col].std()
                df[col] = df[col].apply(lambda x: (x-series_mean)/series_std)
            return df

        scale_columns = ['승', '패', '세', '홀드', '블론', '경기', '선발', '이닝', '삼진/9', '볼넷/9', 
                         '홈런/9', 'BABIP', 'LOB%', 'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR']
        picher_df = standard_scaling(picher, scale_columns)
        picher_df = picher_df.rename(columns={'연봉(2018)': 'y'})
        
        # 팀명 원-핫 인코딩
        team_encoding = pd.get_dummies(picher['팀명'], dtype=int)
        picher_df = picher_df.drop('팀명', axis=1)
        picher_df = picher_df.join(team_encoding)

    with st.expander("피처 간 상관관계 확인"):
        # 피처 간 상관관계 확인
        st.write("피처 간 상관관계 테이블 및 히트맵:")
        corr = picher_df.drop(columns=['선수명']).corr()
        st.write(corr)
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        st.pyplot(plt)

    with st.expander("회귀 분석 모델 학습 및 평가"):
        # train / test 분리
        X = picher_df[picher_df.columns.difference(['선수명', 'y'])]
        y = picher_df['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

        # 회귀 분석 모델 학습
        lr = LinearRegression()
        model = lr.fit(X_train, y_train)
        
        # 다중공선성 확인
        st.write("다중공선성 확인 (VIF):")
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif["features"] = X.columns
        st.write(vif.round(1))

        # 다중공선성이 큰 피처 제거 후 다시 모델 학습
        st.write("다중공선성이 큰 피처를 제거한 후 다시 모델 학습:")
        selected_features = ['FIP', 'WAR', '볼넷/9', '삼진/9', '연봉(2017)']
        X = picher_df[selected_features]
        y = picher_df['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
        model = lr.fit(X_train, y_train)
        
        # 모델 평가
        st.write("모델 평가:")
        st.write("Train R2 score:", model.score(X_train, y_train))
        st.write("Test R2 score:", model.score(X_test, y_test))

        y_train_pred = lr.predict(X_train)
        y_test_pred = lr.predict(X_test)
        st.write("Train RMSE score:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
        st.write("Test RMSE score:", np.sqrt(mean_squared_error(y_test, y_test_pred)))

    with st.expander("최종 회귀 계수 시각화"):
        # 최종 회귀 계수 시각화
        coefs = model.coef_
        coefs_series = pd.Series(coefs)
        x_labels = X.columns
        plt.figure(figsize=(10, 6))
        ax = coefs_series.plot(kind='bar')
        ax.set_title('Feature Coefficients (Selected Features)')
        ax.set_xlabel('Features')
        ax.set_ylabel('Coefficient')
        ax.set_xticklabels(x_labels, rotation=45)
        st.pyplot(plt)

    # 최종 모델을 사용하여 예측값 생성
    st.write("최종 모델을 사용한 예측값 생성:")
    picher_df['예측연봉(2018)'] = model.predict(X).round(0)
    
    # 연봉 단위 통일
    result_df = picher_df[['선수명', 'y', '예측연봉(2018)', '연봉(2017)']]
    result_df = result_df.rename(columns={'y': '실제연봉(2018)'})
    st.write(result_df)

    # 실제 연봉(2018) 기준 상위 Top 10 비교 차트
    st.write("실제 연봉(2018) 기준 상위 Top 10 비교:")
    top10 = result_df.nlargest(10, '실제연봉(2018)')
    top10 = top10.set_index('선수명')
    
    plt.figure(figsize=(12, 8))
    top10[['연봉(2017)', '실제연봉(2018)', '예측연봉(2018)']].plot(kind='bar')
    plt.title('Top 10 실제 연봉(2018), 예측 연봉(2018), 연봉(2017) 비교')
    plt.xlabel('선수명')
    plt.ylabel('연봉')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # 모델 수식 출력
    st.write("### 회귀 모델 수식:")
    intercept = model.intercept_
    coefficients = model.coef_
    equation = f"연봉(2018) = {intercept:.2f}"
    for coef, feature in zip(coefficients, selected_features):
        equation += f" + ({coef:.2f} * {feature})"
    st.write(equation)
    
    # 각 피처의 Top 20 평균값 계산
    top20_mean = picher_df.nlargest(20, 'y')[selected_features].mean()

    # 사이드바에 피처 값 입력
    st.sidebar.header("회귀 모델 시뮬레이션")
    inputs = {}
    for feature in selected_features:
        default_value = top20_mean[feature]
        if feature != '연봉(2017)':
            value = st.sidebar.text_input(f"{feature} 입력값", value=f"{default_value:.2f}")
        else:
            value = st.sidebar.text_input(f"{feature} 입력값 (1000단위)", value=f"{default_value:.0f}")
        inputs[feature] = float(value)
    
    if st.sidebar.button("조회"):
        # 예측 연봉 계산
        prediction = intercept
        for feature in selected_features:
            prediction += coefficients[selected_features.index(feature)] * inputs[feature]
        prediction = round(prediction, 0)
        st.sidebar.write(f"예측 연봉(2018): {prediction} 원")

with st.expander("요약 내용 및 전체 코드"):
    st.divider()
    st.write("""
    데이터 로드 및 초기 탐색을 위해 다음 단계를 수행해줘:
    1. pandas를 사용하여 CSV 파일을 업로드할 수 있는 Streamlit 파일 업로더를 생성해줘.
    2. 업로드된 파일을 데이터프레임으로 읽어와서 데이터 미리보기를 출력해줘.
    3. 데이터프레임의 요약 통계를 출력해줘.
    """)

    st.divider()
    st.write("""
    데이터 정규화 및 전처리를 위해 다음 단계를 수행해줘:
    1. 연봉 데이터를 1000 단위로 정규화해줘 (연봉(2017) 및 연봉(2018)).
    2. 표준화할 피처 목록을 정의하고, 각 피처를 표준화해줘.
    3. 팀명을 원-핫 인코딩으로 변환해줘.
    4. 데이터프레임을 전처리하고 피처 스케일링을 수행한 결과를 출력해줘.
    """)

    st.divider()
    st.write("""
    피처 간 상관관계를 확인하기 위해 다음 단계를 수행해줘:
    1. 데이터프레임에서 선수명을 제외한 나머지 피처들 간의 상관관계 테이블을 생성해줘.
    2. 상관관계 히트맵을 그려줘.
    """)

    st.divider()
    st.write("""
    회귀 분석 모델을 학습하고 평가하기 위해 다음 단계를 수행해줘:
    1. 데이터프레임에서 '선수명'과 'y'를 제외한 나머지 피처들을 독립 변수로 설정하고, 'y'를 종속 변수로 설정해줘.
    2. 학습 데이터와 테스트 데이터로 분할해줘 (80% 학습, 20% 테스트).
    3. 선형 회귀 모델을 학습해줘.
    4. 다중공선성을 확인하기 위해 VIF 값을 계산하고 출력해줘.
    5. VIF 값이 높은 피처들을 제거하고, 최종 피처들을 사용하여 모델을 다시 학습해줘.
    6. 모델의 R2 점수와 RMSE 점수를 학습 데이터와 테스트 데이터에 대해 각각 계산하고 출력해줘.
    """)

    st.divider()
    st.write("""
    회귀 계수를 시각화하고 예측 결과를 생성하기 위해 다음 단계를 수행해줘:
    1. 최종 회귀 모델의 회귀 계수를 막대 그래프로 시각화해줘.
    2. 최종 모델을 사용하여 예측 연봉(2018) 값을 계산해줘.
    3. 선수명, 실제 연봉(2018), 예측 연봉(2018), 연봉(2017)을 포함한 데이터프레임을 생성하고 출력해줘.
    4. 실제 연봉(2018) 기준 상위 10명의 선수에 대해 실제 연봉, 예측 연봉, 연봉(2017)을 비교하는 막대 그래프를 그려줘.
    """)

    st.divider()
    st.write("""
    회귀 모델 수식을 출력하고 사용자 입력을 통해 예측 연봉을 계산하기 위해 다음 단계를 수행해줘:
    1. 회귀 모델의 절편(intercept)과 회귀 계수를 사용하여 회귀 수식을 작성해줘.
    2. 각 피처의 상위 20명의 평균 값을 계산해줘.
    3. 사용자가 사이드바에 피처 값을 입력할 수 있도록 입력 필드를 생성해줘.
    4. 사용자가 입력한 값으로 예측 연봉(2018)을 계산하여 사이드바에 출력해줘.
    """)

        
