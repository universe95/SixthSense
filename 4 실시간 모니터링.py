import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import random

# 페이지 설정
st.set_page_config(
    page_title="실시간 품질 모니터링 시스템",
    page_icon=":경광등:",
    layout="wide"
)

st.title("실시간 품질 모니터링 알람 시스템")
st.markdown("---")

# 임계값 설정 (사이드바)
st.sidebar.header("임계값 설정")
thresholds = {
    '명도수준': st.sidebar.number_input("명도수준 임계값", value=2596.0, format="%.1f"),
    '기준편차': st.sidebar.number_input("기준편차 임계값", value=94.0, format="%.1f"),
    '상대강도': st.sidebar.number_input("상대강도 임계값", value=172.0, format="%.1f"),
    '점형지수': st.sidebar.number_input("점형지수 임계값", value=6.232, format="%.3f"),
    '패치신호': st.sidebar.number_input("패치신호 임계값", value=81.6, format="%.1f")
}

# 결함 유형별 예상 분류
defect_groups = {
    '심각한 결함군': [14, 21, 99],
    '물리적 결함군': [22, 28],
    '일반 결함군': [10, 17, 20, 39, 56]
}

# 실시간 데이터 시뮬레이션 함수
def generate_sample_data():
    """실시간 데이터 시뮬레이션"""
    return {
        '명도수준': random.uniform(500, 4000),
        '기준편차': random.uniform(-5, 800),
        '상대강도': random.uniform(-5, 800),
        '점형지수': random.uniform(0, 15),
        '패치신호': random.uniform(-700, 1200),
        '타임스탬프': datetime.now()
    }

# 위험도 계산 함수
def calculate_risk_level(data, thresholds):
    """5개 특성 기반 위험도 계산"""
    risk_factors = 0
    alerts = []
    
    if data['명도수준'] < thresholds['명도수준']:
        risk_factors += 1
        alerts.append(f"명도수준 위험: {data['명도수준']:.1f} < {thresholds['명도수준']}")
    if data['기준편차'] >= thresholds['기준편차']:
        risk_factors += 1
        alerts.append(f"기준편차 위험: {data['기준편차']:.1f} ≥ {thresholds['기준편차']}")
    if data['상대강도'] >= thresholds['상대강도']:
        risk_factors += 1
        alerts.append(f"상대강도 위험: {data['상대강도']:.1f} ≥ {thresholds['상대강도']}")
    if data['점형지수'] >= thresholds['점형지수']:
        risk_factors += 1
        alerts.append(f"점형지수 위험: {data['점형지수']:.3f} ≥ {thresholds['점형지수']}")
    if data['패치신호'] >= thresholds['패치신호']:
        risk_factors += 1
        alerts.append(f"패치신호 위험: {data['패치신호']:.1f} ≥ {thresholds['패치신호']}")
    
    if risk_factors >= 4:
        level = "HIGH"
        color = "red"
    elif risk_factors >= 2:
        level = "MEDIUM"
        color = "orange"
    elif risk_factors >= 1:
        level = "LOW"
        color = "yellow"
    else:
        level = "NORMAL"
        color = "green"
        
    return level, color, risk_factors, alerts

# 예상 결함 유형 예측
def predict_defect_type(data, thresholds):
    predictions = []
    if (data['명도수준'] < thresholds['명도수준'] and
        data['상대강도'] < thresholds['상대강도'] and
        data['패치신호'] < thresholds['패치신호']):
        predictions.extend([14, 21, 99])
    if (data['기준편차'] >= thresholds['기준편차'] and
        data['상대강도'] >= thresholds['상대강도']):
        predictions.extend([22, 28])
    if not predictions:
        predictions.extend([10, 17, 20, 39, 56])
    return list(set(predictions))

# 메인 화면
top_col1, top_col2 = st.columns([1, 2])

with top_col1:
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: rgba(25, 50, 100, 0.7); /* 연한 남색 + 투명도 */
            color: #ffffff; /* 흰색 글씨 */
            border: none;
            font-weight: bold;
            padding: 0.6em 1.2em;
            border-radius: 6px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.25); /* 입체 그림자 */
            transition: background-color 0.3s ease, 
                        box-shadow 0.3s ease, 
                        transform 0.15s ease; /* 부드러운 애니메이션 */
        }
        div.stButton > button:first-child:hover {
            background-color: rgba(25, 50, 100, 0.85); /* 살짝 진해짐 */
            box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.3);
            transform: translateY(-3px); /* 부드럽게 위로 */
        }
        div.stButton > button:first-child:active {
            background-color: rgba(20, 45, 90, 0.9); /* 더 진한 남색 */
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.2);
            transform: translateY(1px) scale(0.98); /* 살짝 줄어드는 눌림 효과 */
            transition: transform 0.05s ease; /* 클릭 시 빠른 반응 */
        }
        </style>
        """, unsafe_allow_html=True)


    if st.button("데이터 업데이트"):
        if 'monitoring_data' not in st.session_state:
            st.session_state.monitoring_data = []
        new_data = generate_sample_data()
        st.session_state.monitoring_data.append(new_data)
        if len(st.session_state.monitoring_data) > 20:
            st.session_state.monitoring_data = st.session_state.monitoring_data[-20:]



    if 'monitoring_data' in st.session_state and st.session_state.monitoring_data:
        latest_data = st.session_state.monitoring_data[-1]
        st.markdown(f"**측정 시간:** {latest_data['타임스탬프'].strftime('%Y-%m-%d %H:%M:%S')}")

        # 권장 조치 박스 (위험 요소 포함)
        risk_level, color, risk_factors, alerts = calculate_risk_level(latest_data, thresholds)
        if risk_factors >= 4:
            st.markdown(
                f"""
                <div style="background-color: #ffebee; padding: 12px; border-radius: 10px; border: 2px solid #f44336; height: 140px; display: flex; flex-direction: column; justify-content: center;">
                    <h4 style="color: #d32f2f; margin-top: 0;">🚨 긴급 조치 필요</h4>
                    <p style="margin: 0 0 5px 0; color: #333;">위험 요소: {risk_factors}/5</p>
                    <ul style="color: #333; margin-bottom: 0; font-size: 13px;">
                        <li><strong>즉시 생산 중단 검토</strong></li>
                        <li><strong>전수 점검 실시</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True
            )
        elif risk_factors >= 2:
            st.markdown(
                f"""
                <div style="background-color: #fff3e0; padding: 12px; border-radius: 10px; border: 2px solid #ff9800; height: 140px; display: flex; flex-direction: column; justify-content: center;">
                    <h4 style="color: #f57c00; margin-top: 0;">⚠️ 주의 필요</h4>
                    <p style="margin: 0 0 5px 0; color: #333;">위험 요소: {risk_factors}/5</p>
                    <ul style="color: #333; margin-bottom: 0; font-size: 13px;">
                        <li><strong>집중 모니터링 필요</strong></li>
                        <li><strong>샘플링 검사 강화</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True
            )
        elif risk_factors >= 1:
            st.markdown(
                f"""
                <div style="background-color: #e3f2fd; padding: 12px; border-radius: 10px; border: 2px solid #2196f3; height: 140px; display: flex; flex-direction: column; justify-content: center;">
                    <h4 style="color: #1976d2; margin-top: 0;">ℹ️ 관찰 필요</h4>
                    <p style="margin: 0 0 5px 0; color: #333;">위험 요소: {risk_factors}/5</p>
                    <ul style="color: #333; margin-bottom: 0; font-size: 13px;">
                        <li><strong>주의 깊게 관찰</strong></li>
                        <li><strong>주기적 점검 실시</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color: #e8f5e8; padding: 12px; border-radius: 10px; border: 2px solid #4caf50; height: 140px; display: flex; flex-direction: column; justify-content: center;">
                    <h4 style="color: #388e3c; margin-top: 0;">✅ 정상 운영</h4>
                    <p style="margin: 0 0 5px 0; color: #333;">위험 요소: {risk_factors}/5</p>
                    <ul style="color: #333; margin-bottom: 0; font-size: 13px;">
                        <li><strong>정상 운영 지속</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True
            )

with top_col2:
    if 'monitoring_data' in st.session_state and len(st.session_state.monitoring_data) > 1:
        df_history = pd.DataFrame(st.session_state.monitoring_data)
        fig_time = px.line(df_history, x='타임스탬프', y=['명도수준', '기준편차', '상대강도', '점형지수', '패치신호'],
                          title="시계열 변화 추이")
        fig_time.update_layout(height=300)
        st.plotly_chart(fig_time, use_container_width=True)

# 자동 새로고침
auto_refresh = st.sidebar.checkbox("자동 새로고침 (5초)", value=False)
if auto_refresh:
    time.sleep(5)
    st.rerun()
    
st.markdown("---")

# 데이터가 있을 때만
if 'monitoring_data' in st.session_state and st.session_state.monitoring_data:
    latest_data = st.session_state.monitoring_data[-1]
    risk_level, color, risk_factors, alerts = calculate_risk_level(latest_data, thresholds)
    predicted_defects = predict_defect_type(latest_data, thresholds)
    
    features = ['명도수준', '기준편차', '상대강도', '점형지수', '패치신호']
    values = [latest_data[f] for f in features]
    threshold_vals = [thresholds[f] for f in features]
    
    gauge_cols = st.columns(5)
    for i, (feature, value, threshold) in enumerate(zip(features, values, threshold_vals)):
        with gauge_cols[i]:
            if feature == '명도수준' and value < threshold:
                st.markdown('<div style="text-align: center; margin-bottom: 0px;"><strong>🚨 위험</strong></div>', unsafe_allow_html=True)
            elif feature == '기준편차' and value >= threshold:
                st.markdown('<div style="text-align: center; margin-bottom: 0px;"><strong>🚨 위험</strong></div>', unsafe_allow_html=True)
            elif feature == '상대강도' and value >= threshold:
                st.markdown('<div style="text-align: center; margin-bottom: 0px;"><strong>🚨 위험</strong></div>', unsafe_allow_html=True)
            elif feature == '점형지수' and value >= threshold:
                st.markdown('<div style="text-align: center; margin-bottom: 0px;"><strong>🚨 위험</strong></div>', unsafe_allow_html=True)
            elif feature == '패치신호' and value >= threshold:
                st.markdown('<div style="text-align: center; margin-bottom: 0px;"><strong>🚨 위험</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="text-align: center; margin-bottom: 0px;"><strong>✅ 정상</strong></div>', unsafe_allow_html=True)
    
    fig_gauge = go.Figure()
    for i, (feature, value, threshold) in enumerate(zip(features, values, threshold_vals)):
        fig_gauge.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = value,
            domain = {'row': 0, 'column': i},
            title = {'text': feature, 'font': {'color': 'black', 'size': 18}},
            delta = {'reference': threshold},
            gauge = {
                'axis': {'range': [None, max(value*1.2, threshold*1.2)]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, threshold], 'color': "lightgray"},
                    {'range': [threshold, max(value*1.2, threshold*1.2)], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold
                }
            }
        ))
    fig_gauge.update_layout(
        grid = {'rows': 1, 'columns': 5, 'pattern': "independent"},
        height=200,
        margin=dict(t=5, b=0, l=0, r=0)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; margin-top: 10px;">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 18px; margin-right: 8px;">🔍</span>
                <strong style="color: #495057; font-size: 16px;">예상 결함 유형:</strong>
                <span style="margin-left: 10px; color: #007bff; font-weight: 600; font-size: 16px;">{', '.join(map(str, predicted_defects))}</span>
            </div>
        </div>
        """, unsafe_allow_html=True
    )
