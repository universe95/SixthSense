import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import os
from sklearn.metrics import f1_score
import matplotlib.font_manager as fm
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="결함유형 분석 및 예측 시뮬레이터", layout="wide")

# ===================== 공통 설정 =====================
df = pd.read_csv("반도체_결함_데이터_한글.csv")

# ===================== 페이지 제목 =====================
st.title("결함유형 분석 및 예측 시뮬레이터")

# ===================== 탭 생성 =====================
tab1, tab2 = st.tabs(["1. 결함유형 분석 모델", "2. 결함유형 예측 모델"])
# ===================== TAB 1: 결함유형 분석 =====================
with tab1:
    # 1. 데이터 로딩 및 전처리
    @st.cache_data
    def load_data():
        df = pd.read_csv("반도체_결함_데이터_한글.csv")
        df = df.dropna(subset=["결함유형"])
        df["결함유형"] = df["결함유형"].astype(str)
        return df

    # 2. 모델과 최적 임계값을 함께 저장/로드하는 함수
    @st.cache_resource
    def load_or_train_model():
        MODEL_PATH = "defect_model_with_threshold.pkl"
        
        df = load_data()
        X = df.drop(columns=['결함유형', '불량여부'])
        y = df['결함유형']
        
        # 전처리
        X_encoded = pd.get_dummies(X)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # train/test 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
        )
        
        # SMOTE 적용
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                saved_data = pickle.load(f)
            # 임계값을 0.3로 강제 업데이트
            saved_data['threshold'] = 0.3
            return saved_data
        else:
            # 모델 학습
            model = RandomForestClassifier(random_state=42, class_weight='balanced')
            model.fit(X_resampled, y_resampled)
            
            # 임계값을 0.3로 고정 (미분류 줄이기 위함)
            best_threshold = 0.3
            
            # 모델, 임계값, 라벨인코더, 테스트 데이터 저장
            model_data = {
                'model': model,
                'threshold': best_threshold,
                'label_encoder': le,
                'X_test': X_test,
                'y_test': y_test,
                'feature_names': X_encoded.columns.tolist()
            }
            
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(model_data, f)
            
            st.success("🔥 모델을 새로 학습하고 최적 임계값을 계산했습니다.")
            return model_data

    def create_prediction_results(model_data):
        """예측 결과 생성"""
        model = model_data['model']
        threshold = model_data['threshold']
        le = model_data['label_encoder']
        X_test = model_data['X_test']
        y_test = model_data['y_test']
        
        # 예측
        y_proba = model.predict_proba(X_test)
        y_pred_thresh = []
        
        for probs in y_proba:
            max_prob = np.max(probs)
            pred_class = np.argmax(probs)
            if max_prob >= threshold:
                y_pred_thresh.append(pred_class)
            else:
                y_pred_thresh.append(-1)  # 미분류
        
        # 라벨 변환
        pred_labels = []
        for v in y_pred_thresh:
            if v == -1:
                pred_labels.append("미분류")
            else:
                pred_labels.append(le.inverse_transform([v])[0])
        
        actual_labels = le.inverse_transform(y_test)
        
        return pd.DataFrame({
            '실제': actual_labels, 
            '예측': pred_labels,
            '최대확률': [np.max(probs) for probs in y_proba]
        })

    def create_analysis_charts(pred_df, target_label):
        """분석 차트 생성"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 실제 '{target_label}' 예측 분석")
            
            # 해당 라벨의 예측 결과 분석
            mask_target = (pred_df['실제'] == target_label)
            target_data = pred_df[mask_target]
            
            if len(target_data) > 0:
                result_counts = target_data['예측'].value_counts()
                
                # 색상 매핑
                colors = []
                labels_with_counts = []
                for label, count in result_counts.items():
                    if label == target_label:
                        colors.append('#28a745')  # 정분류는 초록색
                    elif label == '미분류':
                        colors.append('#ffc107')  # 미분류는 노란색
                    else:
                        colors.append('#dc3545')  # 오분류는 빨간색
                    labels_with_counts.append(f"{label} ({count}건)")
                
                fig = px.pie(values=result_counts.values, 
                           names=labels_with_counts,
                           color_discrete_sequence=colors)
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(showlegend=True, height=400)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 정확도 표시
                correct_count = result_counts.get(target_label, 0)
                accuracy = correct_count / len(target_data) * 100
                st.metric("정확도", f"{accuracy:.1f}%", f"{correct_count}/{len(target_data)}")
            else:
                st.info("해당 결함유형이 테스트셋에 없습니다.")
        
        with col2:
            st.subheader("정상으로 오분류된 사례")
            
            # 9번으로 잘못 예측된 사례 분석
            misclassified_as_9 = pred_df[(pred_df['예측'] == '9') & (pred_df['실제'] != '9')]
            
            if not misclassified_as_9.empty:
                error_counts = misclassified_as_9['실제'].value_counts()
                labels_with_counts = [f"{label} ({count}건)" for label, count in error_counts.items()]
                
                fig = px.pie(values=error_counts.values, 
                           names=labels_with_counts,
                           color_discrete_sequence=px.colors.qualitative.Set3)
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(showlegend=True, height=400)
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric("총 오분류 건수", len(misclassified_as_9))
            else:
                st.success("✅ 9번으로 오분류된 사례가 없습니다!")

    # 메인 실행부
    df = load_data()
    model_data = load_or_train_model()
    
    # 최적 임계값 표시
    st.success(f"🎯 **임계값: {model_data['threshold']:.1f}** (미분류 최소화를 위해 고정)")
    
    # 예측 결과 생성
    pred_df = create_prediction_results(model_data)
    
    # 전체 성능 지표
    st.markdown("### 📈 전체 모델 성능")
    
    total_samples = len(pred_df)
    classified_samples = len(pred_df[pred_df['예측'] != '미분류'])
    unclassified_samples = total_samples - classified_samples
    
    # 분류된 샘플의 정확도 계산
    classified_mask = pred_df['예측'] != '미분류'
    if classified_samples > 0:
        accuracy = (pred_df[classified_mask]['실제'] == pred_df[classified_mask]['예측']).mean()
    else:
        accuracy = 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f'<div style="font-size: 20px;"><b>전체 테스트 샘플</b><br>{total_samples}</div>', unsafe_allow_html=True)
    col2.markdown(f'<div style="font-size: 20px;"><b>분류된 샘플</b><br>{classified_samples}</div>', unsafe_allow_html=True)
    col3.markdown(f'<div style="font-size: 20px;"><b>미분류 샘플</b><br>{unclassified_samples}</div>', unsafe_allow_html=True)
    col4.markdown(f'<div style="font-size: 20px;"><b>분류 정확도</b><br>{accuracy:.1%}</div>', unsafe_allow_html=True)
    
    # 결함유형 선택 및 분석
    st.markdown("### 🔍 결함유형별 상세 분석")
    target_label = st.selectbox(
        "분석할 결함유형을 선택하세요", 
        options=sorted(df["결함유형"].unique()),
        help="선택한 결함유형의 예측 성능을 자세히 분석합니다"
    )
    
    create_analysis_charts(pred_df, target_label)
            
# ======================TAB 2: 결함유형 예측 모델 ==========================
with tab2:
    # 불량여부 문자열 -> 0/1
    if df['불량여부'].dtype != 'int64' and df['불량여부'].dtype != 'float64':
        df['불량여부'] = df['불량여부'].astype(str).map({'FALSE': 0, 'REAL': 1})

    # 1) 공정단계/공정명 표준화 (공정명_std 사용)
    code2name = {"KB073100": "PC", "KB268900": "RMG", "KB425000": "CBCMP"}
    df["공정명_std"] = df.get("공정명", pd.Series(index=df.index, dtype=object)).copy()
    mask_fill = df["공정명_std"].isna() | (df["공정명_std"] == "")
    if "공정단계" in df.columns:
        df.loc[mask_fill, "공정명_std"] = df.loc[mask_fill, "공정단계"].map(code2name)

    id_cols = ["공정단계","공정명","배치번호","웨이퍼위치","검사순번"]
    target_bin = "불량여부"
    target_mul = "결함유형"

    feature_cols = [
        "공정명_std",
        # 결함 크기
        "가로길이","세로길이","직경크기","검출면적",
        # 광학 특성
        "신호강도","신호극성","에너지값",
        # 위치 정보
        "중심거리","방향각도",
        # 신호 품질
        "기준편차","명도수준","잡음정도",
        # 결함 특성화
        "정렬정도","점형지수","영역잡음","상대강도","활성지수","패치신호",
    ]

    missing = [c for c in feature_cols + [target_bin, target_mul] if c not in df.columns]
    if missing:
        st.error(f"다음 컬럼이 데이터에 없습니다: {missing}")

    # =========================================================
    # 2) 사이드바 입력 UI
    # =========================================================
    X_orig = df[feature_cols].copy()
    num_cols = X_orig.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    st.sidebar.header("시뮬레이션 입력")
    st.sidebar.caption("수치형은 슬라이더, 범주형은 선택으로 설정하세요.")

    def qminmax(s: pd.Series, q=0.01):
        if s.dropna().empty:
            return 0.0, 1.0
        lo = float(s.quantile(q))
        hi = float(s.quantile(1-q))
        if lo == hi:
            lo, hi = float(s.min()), float(s.max())
            if lo == hi:
                lo, hi = lo - 1, hi + 1
        return lo, hi

    user_input = {}
    for c in num_cols:
        lo, hi = qminmax(X_orig[c])
        default = float(np.clip(X_orig[c].median(), lo, hi))
        step = 1.0 if pd.api.types.is_integer_dtype(df[c]) else (hi - lo) / 100.0 or 0.01
        user_input[c] = st.sidebar.slider(c, min_value=float(lo), max_value=float(hi),
                                         value=float(default), step=float(step))

    for c in cat_cols:
        choices = sorted(df[c].dropna().astype(str).unique().tolist())
        if not choices:
            choices = ["(없음)"]
        user_input[c] = st.sidebar.selectbox(c, choices, index=0)

    user_input_df = pd.DataFrame([user_input])

    # 인코딩 유틸
    def encode_align(train_X: pd.DataFrame, single_row_X: pd.DataFrame):
        train_enc = pd.get_dummies(train_X, drop_first=True)
        single_enc = pd.get_dummies(single_row_X, drop_first=True)
        single_enc = single_enc.reindex(columns=train_enc.columns, fill_value=0)
        return train_enc, single_enc
    # =========================================================
    
    y_mul = df[target_mul].astype(str)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_orig, y_mul, test_size=0.2, random_state=42, stratify=y_mul
    )
    X_train, _ = encode_align(X_train_raw, user_input_df)
    X_test = pd.get_dummies(X_test_raw, drop_first=True).reindex(columns=X_train.columns, fill_value=0)
    X_single = pd.get_dummies(user_input_df, drop_first=True).reindex(columns=X_train.columns, fill_value=0)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    with st.spinner("결함유형 모델 학습 중..."):
        model.fit(X_res, y_res)

    y_pred = model.predict(X_test)

    # 예측 결과 계산
    proba = model.predict_proba(X_single)[0]
    classes = model.classes_
    proba_tbl = pd.DataFrame({"결함유형": classes, "예측확률": np.round(proba, 4)})\
                     .sort_values("예측확률", ascending=False).reset_index(drop=True)
    
    # 가장 높은 확률의 결함유형
    top_defect = classes[np.argmax(proba)]
    top_probability = float(np.max(proba))
    
    # 상단: 예측 결함유형과 확률
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 15px; border-radius: 8px; background-color: #f8f9fa;">
                <div style="font-size: 20px; color: #000000; font-weight: bold; margin-bottom: 6px;">예측 결함유형</div>
                <div style="font-size: 28px; font-weight: 800; color: #e91e63;">
                    {top_defect}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 15px; border-radius: 8px; background-color: #f8f9fa;">
                <div style="font-size: 20px; color: #000000; font-weight: bold; margin-bottom: 6px;">예측 확률</div>
                <div style="font-size: 28px; font-weight: 800; color: #e91e63;">
                    {top_probability*100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 본문 2열: 좌(확률 바차트), 우(결함유형별 특성 분석)
    left, right = st.columns([1.2, 0.8])

    # ---- 좌측: 확률 바차트 시각화
    with left:
        st.subheader("결함유형별 예측 확률")
        
        # matplotlib으로 가로 바차트 생성
        fig, ax = plt.subplots(figsize=(5, 4))
        
        # 색상 팔레트 (결함유형별 고정 색상)
        colors = ['#e91e63', '#9c27b0', '#3f51b5', '#2196f3', '#009688', 
                 '#4caf50', '#ff9800', '#ff5722', '#795548', '#607d8b']
        bar_colors = [colors[i % len(colors)] for i in range(len(proba_tbl))]
        
        # 가로 바차트
        bars = ax.barh(range(len(proba_tbl)), proba_tbl['예측확률'], 
                      color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        # y축 라벨 설정
        ax.set_yticks(range(len(proba_tbl)))
        ax.set_yticklabels(proba_tbl['결함유형'], fontsize=10)
        ax.invert_yaxis()  # 가장 높은 확률이 위에 오도록
        
        # x축 설정
        ax.set_xlabel('예측 확률', fontsize=11)
        ax.set_xlim(0, 1.0)
        
        # 확률 값 표시 (바 끝에)
        for i, (idx, row) in enumerate(proba_tbl.iterrows()):
            prob = row['예측확률']
            ax.text(prob + 0.01, i, f'{prob:.3f}', 
                   va='center', fontsize=9, fontweight='bold')
        
        # 스타일링
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_facecolor('#fafafa')
        
        plt.tight_layout()
        st.pyplot(fig)

    # ---- 우측: 결함유형별 특성 분석
    with right:
        st.subheader("결함 특성 분석")
        
        # Top 3 결함유형에 대한 특성 분석
        top3_defects = proba_tbl.head(3)
        
        for idx, row in top3_defects.iterrows():
            defect_type = row['결함유형']
            probability = row['예측확률']
            color = colors[idx % len(colors)]
            
            # 각 결함유형에 대한 설명 카드
            confidence_level = "높음" if probability > 0.7 else "중간" if probability > 0.3 else "낮음"
            confidence_color = "#2e7d32" if probability > 0.7 else "#f57c00" if probability > 0.3 else "#c62828"
            
            st.markdown(
                f"""
                <div style="border: 1px solid {color}; border-radius: 8px; padding: 12px; margin-bottom: 10px; background-color: #fafafa;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="font-weight: bold; color: {color}; font-size: 22px;">
                            {defect_type}
                        </div>
                        <div style="background: {confidence_color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 15px; font-weight: bold;">
                            {confidence_level}
                        </div>
                    </div>
                    <div style="font-size: 18px; margin-bottom: 6px;">
                        확률: <span style="font-weight: bold; color: {color};">{probability:.3f}</span>
                    </div>
                    <div style="background: {color}; height: 6px; border-radius: 3px; width: {probability*100}%; margin-bottom: 8px;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # 결함 크기 분석
        st.markdown("### 🔍 결함 크기 분석")
        
        defect_width = float(user_input.get("가로길이", 0))
        defect_height = float(user_input.get("세로길이", 0))
        defect_area = float(user_input.get("검출면적", 0))
        
        size_analysis = ""
        if defect_area > 1000:
            size_analysis = "🔴 대형 결함 - 주의 필요"
            size_color = "#d32f2f"
        elif defect_area > 100:
            size_analysis = "🟡 중형 결함 - 모니터링 권장"  
            size_color = "#f57c00"
        else:
            size_analysis = "🟢 소형 결함 - 정상 범위"
            size_color = "#2e7d32"
        st.markdown(
            f"""
            <div style="padding: 15px; border-radius: 6px; background-color: #f8f9fa; border-left: 4px solid {size_color};">
                <div style="font-size: 18px; font-weight: bold; color: {size_color};">
                    {size_analysis}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )