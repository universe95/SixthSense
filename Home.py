import streamlit as st

st.set_page_config(page_title="멀티페이지 웹", layout="wide")

st.title("반도체 품질 관리 시스템")

st.markdown(
    "[📎 Tableau 대시보드 열기](https://prod-apnortheast-a.online.tableau.com/t/teamsparta/views/_17548742749860/1_1?:origin=card_share_link)",
    unsafe_allow_html=True
)

image_path = "dashboard.png"

st.image(image_path, use_container_width=True)  


