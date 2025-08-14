import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import os
import matplotlib.font_manager as fm
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


st.set_page_config(page_title="불량유형 분석 및 예측 시뮬레이터", layout="wide")
# ===================== 공통 설정 =====================
@st.cache_data
def load_data():
    df = pd.read_csv("반도체_결함_데이터_한글.csv")
    return df

df = load_data()

st.title("불량유형 분석 및 예측 시뮬레이터")
# ===================== 탭 생성 =====================
tab1, tab2 = st.tabs(["1. 불량여부 분석 모델", "2. 불량여부 예측 모델"])

# ===================== TAB 1: 불량여부 분석 =====================
with tab1:
    df_defect = df.dropna(subset=['불량여부'])
    X = df_defect.drop(columns=['불량여부', '결함유형'])
    y = df_defect['불량여부'].map({'FALSE': 0, 'REAL': 1})
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, stratify=y, random_state=42
    )
            
    
    MODEL_PATH = "randomforest_defect_model.pkl"
    
    def train_and_save():
        with st.spinner('🔄 모델 학습 중입니다... 잠시만 기다려주세요.'):
            best_rf = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                max_depth=None,
                max_features='log2',
                min_samples_leaf=1,
                min_samples_split=2,
                n_estimators=200
            )
            best_rf.fit(X_train, y_train)
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(best_rf, f)
        
        st.success("🔥 **모델 학습 완료 및 저장 완료!**")
        return best_rf
    
    model = None
    
    # 모델 상태 표시
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ **모델 파일이 없습니다. 학습을 시작합니다.**")
        model = train_and_save()
    else:
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            #st.success("✅ **저장된 최적 모델을 성공적으로 불러왔습니다.**")
        except Exception as e:
            st.error(f"❌ **모델 파일 로드 중 오류 발생:** {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            model = train_and_save()
    
    
    # 모델 성능 평가 섹션
    if model is not None:       
        # Threshold 설정 영역
        with st.container():
            st.markdown("#### ⚙️ 임계값 설정")
            threshold_col1, threshold_col2 = st.columns([3, 1])
            
            with threshold_col1:
                threshold = st.slider(
                    "**Threshold 값 설정 (불량으로 예측할 최소 확률)**", 
                    0.0, 1.0, 0.5, 0.01,
                    help="높은 값: 더 보수적 예측, 낮은 값: 더 민감한 예측"
                )
            
            with threshold_col2:
                st.markdown("**현재 설정값**")
                st.markdown(f"<h2 style='color: #667eea; text-align: center;'>{threshold:.2f}</h2>", 
                           unsafe_allow_html=True)
        
        # 예측 및 성능 계산
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # 핵심 성능 지표 표시
        st.markdown("#### 📈 핵심 성능 지표")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        recall_1 = report['1']['recall']
        recall_0 = report['0']['recall']
        roc_auc = roc_auc_score(y_test, y_proba)


        with metric_col1:
            st.metric(
                label="Recall (불량)",
                value=f"{recall_1:.3f}",
                delta=f"{(recall_1-0.5):.4f}" if recall_1 > 0.5 else f"{(recall_1-0.5):.4f}"
            )

        with metric_col2:
            st.metric(
                label="Recall (정상)",
                value=f"{recall_0:.3f}"
            )

        with metric_col3:
            st.metric(
                label="ROC AUC",
                value=f"{roc_auc:.3f}"
            )

        with metric_col4:
            accuracy = report['accuracy']
            st.metric(
                label="전체 정확도",
                value=f"{accuracy:.3f}"
            )

    
        
       
        st.markdown("---")
        
        # 시각화 섹션
        st.markdown("#### 📊 성능 시각화")
        
        col_cm, col_roc = st.columns(2)
        
        with col_cm:
            with st.container():
                st.markdown("##### 🔍 혼동 행렬")
                fig_cm, ax = plt.subplots(figsize=(6, 5))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                            cbar_kws={'label': '개수'},
                            annot_kws={'size': 16, 'weight': 'bold'})  # 여기에 폰트 크기/굵기 설정 추가
                ax.set_xlabel("예측값", fontsize=12, fontweight='bold')
                ax.set_ylabel("실제값", fontsize=12, fontweight='bold')
                ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                st.pyplot(fig_cm)

                
       
        with col_roc:
            with st.container():
                st.markdown("##### 📈 ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = roc_auc_score(y_test, y_proba)
                
                fig_roc, ax = plt.subplots(figsize=(6, 5))
                ax.plot(fpr, tpr, linewidth=3, label=f"ROC Curve (AUC = {roc_auc:.4f})", color='#667eea')
                ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.8, label='Random Classifier')
                ax.fill_between(fpr, tpr, alpha=0.1, color='#667eea')
                ax.set_xlabel("False Positive Rate", fontsize=12, fontweight='bold')
                ax.set_ylabel("True Positive Rate", fontsize=12, fontweight='bold')
                ax.set_title("ROC Curve", fontsize=14, fontweight='bold', pad=20)
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_roc)
                
        
        st.markdown("---")
        
        # 특성 중요도 섹션
        st.markdown("#### 📌 주요 특성 중요도 (Top 20)")
        with st.container():
            # 특성 중요도 내림차순 정렬 후 상위 20개 선택
            feat_importance = pd.Series(model.feature_importances_, index=X_encoded.columns).sort_values(ascending=False)[:20]

            fig_imp, ax = plt.subplots(figsize=(12, 8))

            # 그래프는 내림차순 순서 그대로 표시(가장 큰 값이 위쪽)
            # 색상 지정: 위에서부터(가장 큰 값 순) 5개는 진한 남색, 나머지는 연한 하늘색
            colors = ['#003366' if i < 5 else '#99ccff' for i in range(len(feat_importance))]
            
            # y축 위치를 0~19로 두고, 그에 맞게 레이블과 값 표시
            bars = ax.barh(range(len(feat_importance)-1, -1, -1), feat_importance.values, color=colors)

    

            # 위에서부터 색상 입히기 위해 막대별 색 지정
            for i, bar in enumerate(bars):
                bar.set_color(colors[i])

            ax.set_yticks(range(len(feat_importance)))
            ax.set_yticklabels(feat_importance.index[::-1], fontsize=10)  # y축 라벨도 역순으로 (가장 큰 게 위)

            ax.set_xlabel("중요도 점수", fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            # 중요도 값을 막대 옆에 표시 (역순 y좌표에 맞게)
            for i, value in enumerate(feat_importance.values[::-1]):
                ax.text(value + 0.001, i, f'{value:.4f}', va='center', fontsize=8, fontweight='bold')

            st.pyplot(fig_imp)
# ======================TAB 2: 불량여부 예측 모델 ==========================
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
    y_bin = df[target_bin].copy()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_orig, y_bin, test_size=0.2, stratify=y_bin, random_state=42
    )
    X_train, _ = encode_align(X_train_raw, user_input_df)
    X_test = pd.get_dummies(X_test_raw, drop_first=True).reindex(columns=X_train.columns, fill_value=0)
    X_single = pd.get_dummies(user_input_df, drop_first=True).reindex(columns=X_train.columns, fill_value=0)

    best_rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        max_depth=None,
        max_features='log2',
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=200
    )

    with st.spinner("불량여부 모델 학습 중..."):
        best_rf.fit(X_train, y_train)

    # 예측
    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:, 1]
    p_single = float(best_rf.predict_proba(X_single)[:, 1][0])
    label_single = "불량" if p_single >= 0.5 else "정상"
    label_color = "#d32f2f" if label_single == "불량" else "#1976d2"  # 빨강/파랑

    # 상단: 예측 불량 확률과 예측 라벨 (컴팩트한 크기)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 15px; border-radius: 8px; background-color: #f8f9fa;">
                <div style="font-size: 20px; color: #000000; font-weight: bold; margin-bottom: 6px;">예측 불량 확률</div>
                <div style="font-size: 28px; font-weight: 800; color: #333;">
                    {p_single*100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 15px; border-radius: 8px; background-color: #f8f9fa;">
                <div style="font-size: 20px; color: #000000; font-weight: bold; margin-bottom: 6px;">예측 라벨</div>
                <div style="font-size: 28px; font-weight: 800; color: {label_color};">
                    {label_single}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 본문 2열: 좌(웨이퍼맵), 우(5개 지표 정상/경고) - 비율 조정
    left, right = st.columns([1.0, 1.0])

    # ---- 좌측: 웨이퍼 맵 (중심거리/방향각도 -> x,y), 결함 형태 반영
    with left:
        st.subheader("웨이퍼 맵")
        needed_for_map = ["중심거리", "방향각도"]
        if not all(c in user_input for c in needed_for_map):
            st.info("웨이퍼 맵을 그리려면 '중심거리'와 '방향각도' 입력이 필요합니다.")
        else:
            r = float(user_input["중심거리"])
            theta_deg = float(user_input["방향각도"])
            theta = np.deg2rad(theta_deg)

            # 반경 정규화: 훈련데이터의 중심거리 최대값 기준
            r_max = float(np.nanmax(X_train_raw["중심거리"])) if "중심거리" in X_train_raw.columns else max(r, 1.0)
            rn = np.clip(r / (r_max + 1e-8), 0, 1)

            x = rn * np.cos(theta)
            y = rn * np.sin(theta)

            # 크기 가중치 계산 (훈련 데이터 기반)
            size_features = ['가로길이', '세로길이', '직경크기', '검출면적']
            size_weights = {}
            
            for col in size_features:
                if col in X_train_raw.columns:
                    max_val = float(X_train_raw[col].max())
                    if max_val > 0:
                        size_weights[col] = max_val
                    else:
                        size_weights[col] = 1.0
                else:
                    size_weights[col] = 1.0

            # 사용자 입력값에 대한 크기 가중치 계산
            user_size_scores = []
            for col in size_features:
                if col in user_input:
                    val = float(user_input[col])
                    normalized_val = val / size_weights[col]
                    user_size_scores.append(normalized_val)
                else:
                    user_size_scores.append(0.0)
            
            size_score = np.mean(user_size_scores)
            marker_size = size_score * 800 + 100  # 기본 크기 + 가중치

            # 결함 크기 정보 수집
            defect_width = float(user_input.get("가로길이", 0))
            defect_height = float(user_input.get("세로길이", 0))
            defect_diameter = float(user_input.get("직경크기", 0))
            defect_area = float(user_input.get("검출면적", 0))

            # 결함 형태 결정 및 크기 정규화 - 웨이퍼 맵 크기 조정
            fig, ax = plt.subplots(figsize=(2.5, 2.5))  # 2.5에서 3.2로 키움
            
            # 웨이퍼 외곽 (단위 원)
            wafer = plt.Circle((0, 0), 1.0, color="#f2f2f2", ec="#999", lw=1.5)
            ax.add_artist(wafer)
            
            # 축 설정
            ax.set_aspect('equal', 'box')
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.axis('off')

            # 보조 그리드(동심원/십자선)
            for rr in [0.25, 0.5, 0.75, 1.0]:
                ax.add_artist(plt.Circle((0, 0), rr, color="#ddd", fill=False, ls='--', lw=0.7))
            ax.plot([-1, 1], [0, 0], ls='--', lw=0.7, color="#ddd")
            ax.plot([0, 0], [-1, 1], ls='--', lw=0.7, color="#ddd")

            # 결함 형태 그리기 (크기 가중치 반영)
            defect_color = '#d32f2f' if label_single == "불량" else '#1976d2'
            
            # 크기 가중치를 반영한 스케일 팩터
            base_scale = 0.08
            weight_adjusted_scale = base_scale * (1 + size_score * 2)  # 가중치에 따라 크기 조정
            
            if defect_diameter > 0:
                # 원형 결함
                radius = weight_adjusted_scale * np.sqrt(defect_diameter / (size_weights["직경크기"] + 1e-8))
                radius = min(max(radius, 0.02), 0.12)  # 최소/최대 크기 설정
                defect_shape = plt.Circle((x, y), radius, color=defect_color, alpha=0.8, edgecolor='white', linewidth=1.5)
                ax.add_artist(defect_shape)
                
            elif defect_width > 0 and defect_height > 0:
                # 직사각형 결함
                w = weight_adjusted_scale * np.sqrt(defect_width / (size_weights["가로길이"] + 1e-8))
                h = weight_adjusted_scale * np.sqrt(defect_height / (size_weights["세로길이"] + 1e-8))
                w = min(max(w, 0.02), 0.12)  # 최소/최대 크기 설정
                h = min(max(h, 0.02), 0.12)
                
                rect = patches.Rectangle((x-w/2, y-h/2), w, h, 
                                        color=defect_color, alpha=0.8, 
                                        edgecolor='white', linewidth=1.5)
                ax.add_patch(rect)
                
            elif defect_area > 0:
                # 면적 기반 원형 결함
                radius = weight_adjusted_scale * np.sqrt(defect_area / (size_weights["검출면적"] + 1e-8))
                radius = min(max(radius, 0.02), 0.12)  # 최소/최대 크기 설정
                defect_shape = plt.Circle((x, y), radius, color=defect_color, alpha=0.8, edgecolor='white', linewidth=1.5)
                ax.add_artist(defect_shape)
            else:
                # 기본 점 마커 (가중치 반영)
                scatter_size = min(max(marker_size, 200), 600)
                ax.scatter([x], [y], s=scatter_size, c=defect_color, alpha=0.85, edgecolor='white', linewidth=2.0)

            st.pyplot(fig)

    # ---- 우측: 5개 핵심 지표 정상/경고 배지 (예측 라벨 박스와 맞춤 정렬)
    with right:
        # 핵심 지표 박스를 오른쪽 정렬하기 위해, 좁은 컬럼을 만들고 그 안에 내용을 넣음
        st.subheader("핵심 지표 상태")
        right_align_col, right_col_badges = st.columns([0.2, 0.8])
        
        with right_col_badges:
            # 임계값 자동 결정 (명도수준: low 위험 → 하위 5% / 나머지 high 위험 → 상위 95%)
            def q(dfcol, qv):
                try:
                    return float(np.nanquantile(dfcol, qv))
                except Exception:
                    return None

            thresholds = {
                "명도수준_low_q05": q(X_train_raw["명도수준"], 0.05) if "명도수준" in X_train_raw.columns else None,
                "기준편차_high_q95": q(X_train_raw["기준편차"], 0.95) if "기준편차" in X_train_raw.columns else None,
                "상대강도_high_q95": q(X_train_raw["상대강도"], 0.95) if "상대강도" in X_train_raw.columns else None,
                "점형지수_high_q95": q(X_train_raw["점형지수"], 0.95) if "점형지수" in X_train_raw.columns else None,
                "패치신호_high_q95": q(X_train_raw["패치신호"], 0.95) if "패치신호" in X_train_raw.columns else None,
            }

            def badge(label, value, status, threshold_desc):
                color = "#2e7d32" if status == "정상" else "#c62828"
                st.markdown(
                    f"""
                    <div style="
                        border:1px solid #e0e0e0; border-radius:6px; padding:10px 12px; margin-bottom:8px;
                        display:flex; flex-direction:column; background-color:#fafafa; max-width: 300px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <div style="font-weight:600; font-size:14px;">{label}</div>
                            <span style="background:{color}; color:white; padding:3px 10px; border-radius:12px; font-weight:700; font-size:11px;">
                                {status}
                            </span>
                        </div>
                        <div style="color:#333; font-size:15px; font-weight:600; margin-bottom:4px;">값: {value}</div>
                        <div style="color:#666; font-size:11px; line-height:1.2;">
                            {threshold_desc}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # 각 지표 판정
            # 명도수준: 낮으면 위험
            if "명도수준" in user_input and thresholds["명도수준_low_q05"] is not None:
                val = float(user_input["명도수준"])
                thr = thresholds["명도수준_low_q05"]
                status = "경고" if val < thr else "정상"
                badge("명도수준", f"{val:.3f}", status, f"기준: 하위 5% ({thr:.3f}) 미만이면 경고")
            # 기준편차: 높으면 위험
            if "기준편차" in user_input and thresholds["기준편차_high_q95"] is not None:
                val = float(user_input["기준편차"])
                thr = thresholds["기준편차_high_q95"]
                status = "경고" if val >= thr else "정상"
                badge("기준편차", f"{val:.3f}", status, f"기준: 상위 95% ({thr:.3f}) 이상이면 경고")
            # 상대강도
            if "상대강도" in user_input and thresholds["상대강도_high_q95"] is not None:
                val = float(user_input["상대강도"])
                thr = thresholds["상대강도_high_q95"]
                status = "경고" if val >= thr else "정상"
                badge("상대강도", f"{val:.3f}", status, f"기준: 상위 95% ({thr:.3f}) 이상이면 경고")
            # 점형지수
            if "점형지수" in user_input and thresholds["점형지수_high_q95"] is not None:
                val = float(user_input["점형지수"])
                thr = thresholds["점형지수_high_q95"]
                status = "경고" if val >= thr else "정상"
                badge("점형지수", f"{val:.3f}", status, f"기준: 상위 95% ({thr:.3f}) 이상이면 경고")
            # 패치신호
            if "패치신호" in user_input and thresholds["패치신호_high_q95"] is not None:
                val = float(user_input["패치신호"])
                thr = thresholds["패치신호_high_q95"]
                status = "경고" if val >= thr else "정상"
                badge("패치신호", f"{val:.3f}", status, f"기준: 상위 95% ({thr:.3f}) 이상이면 경고")