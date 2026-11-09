"""Streamlit entry point: streamlit run app.py"""

import matplotlib.pyplot as plt
import streamlit as st

from qpcr_analysis import QPCRInputError, analyze, read_qpcr_csv


st.set_page_config(page_title="qPCR 상대 발현량 분석", layout="wide")
st.title("qPCR 상대 발현량 분석")
st.write("시료별 기술 반복 Ct를 평균낸 뒤 ΔCt, ΔΔCt, 2^-ΔΔCt를 계산합니다.")
st.caption("필수 CSV 열: Sample, Condition, Gene, Replicate, Ct. Sample은 생물학적 시료마다 고유해야 합니다.")

uploaded = st.file_uploader("CSV 파일", type="csv")
if uploaded is not None:
    try:
        data = read_qpcr_csv(uploaded.getvalue())
        if not {"Sample", "Condition", "Gene", "Replicate", "Ct"}.issubset(data.columns):
            raise QPCRInputError("필수 열이 없습니다: Sample, Condition, Gene, Replicate, Ct")
        genes = sorted(data["Gene"].dropna().astype(str).unique())
        conditions = sorted(data["Condition"].dropna().astype(str).unique())
        if not genes or not conditions:
            raise QPCRInputError("Gene 또는 Condition 값이 없습니다.")
        reference = st.selectbox("기준 유전자", genes)
        default_control = next((i for i, name in enumerate(conditions) if name.lower() in {"mock", "control", "untreated"}), 0)
        control = st.selectbox("대조군 조건", conditions, index=default_control)
        st.info(f"현재 대조군: {control} — 그래프에서 이 조건의 기하평균이 1입니다. 분석 전에 확인하세요.")
        max_sd = st.number_input("기술 반복 Ct SD 허용 상한", min_value=0.01, value=0.30, step=0.05)
        min_reps = st.number_input("시료·유전자별 최소 기술 반복 수", min_value=2, value=2, step=1)
        st.dataframe(data.head(20), hide_index=True)

        if st.button("분석 실행", type="primary"):
            result = analyze(data, reference, control, max_sd, int(min_reps))
            st.success("입력 및 기술 반복 QC 통과")
            for warning in result.warnings:
                st.warning(warning)
            st.subheader("조건별 요약")
            st.dataframe(result.summary, hide_index=True)
            st.subheader("기준 유전자 Ct")
            st.dataframe(result.reference_summary, hide_index=True)
            st.subheader("생물학적 시료별 결과")
            st.dataframe(result.sample_results, hide_index=True)
            st.subheader("기술 반복 QC")
            st.dataframe(result.technical_qc, hide_index=True)

            fig, ax = plt.subplots(figsize=(9, 5))
            groups = result.sample_results.groupby(["Gene", "Condition"], sort=True)
            labels = []
            for i, ((gene, condition), group) in enumerate(groups):
                ax.scatter([i] * len(group), group["Fold_Change"], alpha=0.7)
                geo_mean = result.summary.loc[
                    (result.summary["Gene"] == gene) & (result.summary["Condition"] == condition),
                    "Geometric_Mean_FC",
                ].iloc[0]
                ax.scatter(i, geo_mean, marker="_", s=450, linewidths=3, color="black")
                labels.append(f"{gene}\n{condition}")
            ax.set_xticks(range(len(labels)), labels)
            ax.set_ylabel("Fold change (2^-ΔΔCt)")
            ax.axhline(1, linestyle="--", color="gray")
            ax.set_yscale("log", base=2)
            ax.set_title("점: 생물학적 시료, 검은 선: 기하평균")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            csv = result.sample_results.to_csv(index=False).encode("utf-8-sig")
            st.download_button("시료별 결과 CSV 다운로드", csv, "qpcr_sample_results.csv", "text/csv")
            summary_csv = result.summary.to_csv(index=False).encode("utf-8-sig")
            st.download_button("조건별 요약 CSV 다운로드", summary_csv, "qpcr_summary.csv", "text/csv")
    except QPCRInputError as exc:
        st.error(str(exc))
