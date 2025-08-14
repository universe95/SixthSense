import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import koreanize_matplotlib
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("반도체_결함_데이터_한글.csv")
    df['불량여부'] = df['불량여부'].map({'REAL': 1, 'FALSE': 0})
    return df

df = load_data()


# 사이드바 필터
with st.sidebar:
    st.header("🔍 데이터 필터")

    selected_process = st.selectbox('공정명 선택', ['전체'] + sorted(df['공정명'].dropna().unique().tolist()))
    filtered_df = df if selected_process == '전체' else df[df['공정명'] == selected_process]

    selected_defect = st.selectbox('결함유형 선택', ['전체'] + sorted(filtered_df['결함유형'].dropna().unique().tolist()))
    filtered_df = filtered_df if selected_defect == '전체' else filtered_df[filtered_df['결함유형'] == selected_defect]

    selected_lot = st.selectbox('배치번호 선택', ['전체'] + sorted(filtered_df['배치번호'].dropna().unique().tolist()))
    filtered_df = filtered_df if selected_lot == '전체' else filtered_df[filtered_df['배치번호'] == selected_lot]

    selected_position = st.selectbox('웨이퍼 위치 선택', ['전체'] + sorted(filtered_df['웨이퍼위치'].dropna().unique().tolist()))
    filtered_df = filtered_df if selected_position == '전체' else filtered_df[filtered_df['웨이퍼위치'] == selected_position]


# 최종 필터링 결과
final_df = filtered_df.copy()

st.title("웨이퍼 결함 시각화")
# 결함 개수 출력
st.write(f"선택된 조건에 해당하는 결함 개수: **{len(final_df)}건**")

st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: rgba(25, 50, 100, 0.7); /* 연한 남색 + 투명도 */
            color: #FFFFFF; /* 흰색 글씨 */
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
        
# ▶️ 시각화 버튼
if st.button("📊 시각화", type="primary"):
    if final_df.empty:
        st.warning("조건에 맞는 데이터가 없습니다.")
    else:
        defect_df = final_df.copy()
        
        if defect_df.empty:
            st.warning("불량 결함 데이터가 없습니다.")
        else:
            # 탭 생성
            tab1, tab2, tab3 = st.tabs(["🎯 결함 분포", "🔥 결함 빈도", "📊 결함 밀도"])

            with tab1:
                
                # 좌표 변환
                defect_df['rad'] = np.deg2rad(defect_df['방향각도'])
                defect_df['x'] = defect_df['중심거리'] * np.cos(defect_df['rad'])
                defect_df['y'] = defect_df['중심거리'] * np.sin(defect_df['rad'])

                # 크기 가중치 계산 (더 부드럽게)
                features = ['가로길이', '세로길이', '직경크기', '검출면적']
                for col in features:
                    if col in defect_df.columns:
                        defect_df[f'{col}_norm'] = defect_df[col] / defect_df[col].max()
                
                valid_features = [f'{col}_norm' for col in features if f'{col}_norm' in defect_df.columns]
                if valid_features:
                    defect_df['size_score'] = defect_df[valid_features].mean(axis=1)
                    sizes = defect_df['size_score'] * 30 + 10  # 크기 범위 조정
                else:
                    sizes = 20  # 기본 크기

                # 고정된 결함유형별 색상 매핑 (더 선명한 색상)
                all_defect_types = sorted(df['결함유형'].dropna().unique())
                colors_list = ["#4196FF", "#F7FF01", "#4BFFC9", "#04C269", "#FFD60B", 
                             "#FF5252CE", "#FF69B2", "#DA4CF9", "#FFCDCD", "#FF7F07", "#73FF00"]
                color_dict = {typ: colors_list[i % len(colors_list)] for i, typ in enumerate(all_defect_types)}

                colors = defect_df['결함유형'].map(color_dict)

                # 웨이퍼 반경 설정
                wafer_radius = 150000

                # 컬럼 레이아웃으로 차트와 범례 분리
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
                    
                    # 산점도 그리기
                    scatter = ax.scatter(defect_df['x'], defect_df['y'], 
                                       c=colors, s=sizes, alpha=0.7, 
                                       edgecolors='white', linewidth=0.5)

                    # 웨이퍼 테두리 (더 깔끔하게)
                    circle = plt.Circle((0, 0), wafer_radius, color="#000000", 
                                      fill=False, linewidth=2, alpha=0.8)
                    ax.add_artist(circle)

                    # 기본 설정
                    ax.set_xlim(-wafer_radius*1.05, wafer_radius*1.05)
                    ax.set_ylim(-wafer_radius*1.05, wafer_radius*1.05)
                    ax.set_aspect('equal')
                    ax.set_title("웨이퍼 결함 위치 분포", fontsize=14, fontweight='bold', pad=20)
                    ax.set_xlabel("X Position (μm)", fontsize=10)
                    ax.set_ylabel("Y Position (μm)", fontsize=10)
                    
                    # 격자 제거하고 배경색 설정
                    ax.set_facecolor("#FCFCFC")
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    # 축 눈금 간소화
                    ax.set_xticks([-100000, 0, 100000])
                    ax.set_yticks([-100000, 0, 100000])
                    ax.tick_params(labelsize=8)

                    st.pyplot(fig, bbox_inches='tight')

                with col2:
                    st.write("**결함유형 범례**")
                    filtered_types = defect_df['결함유형'].unique()
                    for typ in filtered_types:
                        count = len(defect_df[defect_df['결함유형'] == typ])
                        color = color_dict[typ]
                        st.markdown(f'<div style="display: flex; align-items: center; margin: 5px 0;">'
                                  f'<div style="width: 15px; height: 15px; background-color: {color}; '
                                  f'border-radius: 50%; margin-right: 8px;"></div>'
                                  f'<span style="font-size: 12px;">{typ} ({count}개)</span></div>', 
                                  unsafe_allow_html=True)
            with tab2:              
                    # matplotlib으로 히트맵 생성
                    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
                    
                    # 2D 히스토그램으로 히트맵 생성
                    hist, xedges, yedges = np.histogram2d(
                        defect_df['x'], defect_df['y'], 
                        bins=80, 
                        range=[[-wafer_radius, wafer_radius], [-wafer_radius, wafer_radius]]
                    )
                    
                    # 히트맵 그리기
                    extent = [-wafer_radius, wafer_radius, -wafer_radius, wafer_radius]
                    im = ax.imshow(hist.T, extent=extent, origin='lower', 
                                 cmap='YlOrRd', alpha=0.8, aspect='equal')
                    
                    # 웨이퍼 테두리
                    circle = plt.Circle((0, 0), wafer_radius, color="#000000", 
                                      fill=False, linewidth=2, alpha=0.8)
                    ax.add_artist(circle)

                    # 기본 설정
                    ax.set_xlim(-wafer_radius*1.05, wafer_radius*1.05)
                    ax.set_ylim(-wafer_radius*1.05, wafer_radius*1.05)
                    ax.set_aspect('equal')
                    ax.set_title("웨이퍼 결함 빈도 히트맵", fontsize=14, fontweight='bold', pad=20)
                    ax.set_xlabel("X Position (μm)", fontsize=10)
                    ax.set_ylabel("Y Position (μm)", fontsize=10)
                    
                    # 격자 제거하고 배경색 설정
                    ax.set_facecolor("#FCFCFC")
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    # 축 눈금 간소화
                    ax.set_xticks([-100000, 0, 100000])
                    ax.set_yticks([-100000, 0, 100000])
                    ax.tick_params(labelsize=8)
                    
                    # 컬러바 추가
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Count', fontsize=10)

                    st.pyplot(fig, bbox_inches='tight')

            with tab3:
                if len(defect_df) > 1:  # KDE는 최소 2개 점이 필요
                    # KDE 기반 밀도 계산
                    x = defect_df['x'].values
                    y = defect_df['y'].values
                    
                    xy = np.vstack([x, y])
                    kde = gaussian_kde(xy)

                    # 시각화용 grid 설정 (해상도 조정)
                    num_bins = 150
                    xgrid = np.linspace(-wafer_radius, wafer_radius, num_bins)
                    ygrid = np.linspace(-wafer_radius, wafer_radius, num_bins)
                    xmesh, ymesh = np.meshgrid(xgrid, ygrid)
                    positions = np.vstack([xmesh.ravel(), ymesh.ravel()])
                    density = kde(positions).reshape(xmesh.shape)

                    # 원형 마스크 생성
                    mask = xmesh**2 + ymesh**2 <= wafer_radius**2
                    density_masked = np.where(mask, density, np.nan)
                    
                    # matplotlib으로 밀도 히트맵 생성
                    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
                        
                    # 원형 마스크를 적용한 밀도 데이터 시각화
                    im = ax.imshow(density_masked, extent=[-wafer_radius, wafer_radius, -wafer_radius, wafer_radius],
                                     origin='lower', cmap='YlOrRd', alpha=0.8, aspect='equal')
                        
                    # 웨이퍼 테두리
                    circle = plt.Circle((0, 0), wafer_radius, color="#000000", 
                                          fill=False, linewidth=2, alpha=0.8)
                    ax.add_artist(circle)

                    # 기본 설정
                    ax.set_xlim(-wafer_radius*1.05, wafer_radius*1.05)
                    ax.set_ylim(-wafer_radius*1.05, wafer_radius*1.05)
                    ax.set_aspect('equal')
                    ax.set_title("웨이퍼 결함 밀도 분포", fontsize=14, fontweight='bold', pad=20)
                    ax.set_xlabel("X Position (μm)", fontsize=10)
                    ax.set_ylabel("Y Position (μm)", fontsize=10)
                        
                    # 격자 제거하고 배경색 설정
                    ax.set_facecolor("#FCFCFC")
                    ax.grid(True, alpha=0.3, linestyle='--')
                        
                    # 축 눈금 간소화
                    ax.set_xticks([-100000, 0, 100000])
                    ax.set_yticks([-100000, 0, 100000])
                    ax.tick_params(labelsize=8)
                        
                    # 컬러바 추가
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Density', fontsize=10)

                    st.pyplot(fig, bbox_inches='tight')
                else:
                    st.warning("밀도 계산을 위해 최소 2개 이상의 결함 데이터가 필요합니다.")



