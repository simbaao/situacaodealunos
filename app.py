# ============================================================
# PREVISÃO DA SITUAÇÃO DE ALUNOS
# Árvore de Decisão + Avaliação + GridSearchCV + Streamlit (tema escuro)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURAÇÃO DA PÁGINA + CSS (TEMA ESCURO)
# ============================================================

st.set_page_config(
    page_title="Previsão da Situação do Aluno",
    page_icon="🎓",
    layout="wide",
)

CSS_EXTRA = """
<style>
.stApp {
    background: #000000;
}

/* Texto padrão em branco */
.stApp, .stMarkdown, p, span, label, div, h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

#cabecalho {
    background: linear-gradient(90deg, #7c3aed 0%, #a855f7 60%, #c084fc 100%);
    padding: 28px 32px;
    border-radius: 20px;
    color: white !important;
    margin-bottom: 24px;
}
#cabecalho h1, #cabecalho p {
    color: white !important;
}

section[data-testid="stSidebar"] {
    background: #0a0a0a;
    border-right: 1px solid #333333;
}

div.stButton > button {
    background: linear-gradient(90deg, #7c3aed 0%, #a855f7 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6em 1.2em;
    font-weight: 600;
    width: 100%;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #6d28d9 0%, #9333ea 100%);
    color: white;
}

div[data-testid="stNumberInput"] input {
    background-color: #1a1a1a;
    border: 1px solid #444444;
    color: #ffffff !important;
    border-radius: 10px;
}

/* Expander e outros containers com fundo escuro */
div[data-testid="stExpander"] {
    background-color: #0a0a0a;
    border: 1px solid #333333;
    border-radius: 10px;
}

.block-container {
    max-width: 980px;
    padding-top: 1.5rem;
}

#rodape {
    text-align: center;
    color: #c084fc !important;
    font-size: 13px;
    margin-top: 20px;
}
</style>
"""
st.markdown(CSS_EXTRA, unsafe_allow_html=True)


# ============================================================
# 1. DADOS + 2-11. TREINO, AVALIAÇÃO, GRÁFICOS (CACHEADO)
# ============================================================

# Cor associada a cada classe, usada nos elementos visuais da interface
CORES_SITUACAO = {
    "Aprovado": "#22c55e",
    "Recuperação": "#f59e0b",
    "Reprovado": "#ef4444",
}

features_numericas = ["Horas_de_estudo", "Faltas", "Nota"]


@st.cache_resource(show_spinner="Treinando e otimizando o modelo...")
def treinar_modelo():
    np.random.seed(42)

    N_ALUNOS = 500

    horas_estudo = np.random.uniform(0, 15, N_ALUNOS)

    faltas = np.random.poisson(lam=7, size=N_ALUNOS)
    faltas = np.clip(faltas, 0, 30)

    ruido = np.random.normal(0, 0.7, N_ALUNOS)

    nota = (
        2.5
        + 0.55 * horas_estudo
        - 0.08 * faltas
        + ruido
    )

    nota = np.clip(nota, 0, 10)
    nota = np.round(nota, 1)

    situacao = np.select(
        [
            nota >= 7,
            nota >= 5
        ],
        [
            "Aprovado",
            "Recuperação"
        ],
        default="Reprovado"
    )

    df = pd.DataFrame({
        "Horas_de_estudo": horas_estudo.round(1),
        "Faltas": faltas,
        "Nota": nota,
        "Situacao": situacao
    })

    # ---- Features (X) e target (y) ----
    X = df[["Horas_de_estudo", "Faltas", "Nota"]]
    y = df["Situacao"]

    # ---- Divisão treino / teste ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ---- Pré-processamento ----
    features_categoricas = []

    preprocessador = ColumnTransformer(
        transformers=[
            ("numericas", "passthrough", features_numericas),
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), features_categoricas)
        ],
        remainder="drop"
    )

    # ---- Modelo base ----
    modelo_base = Pipeline(
        steps=[
            ("preprocessamento", preprocessador),
            ("modelo", DecisionTreeClassifier(random_state=42))
        ]
    )

    # ---- GridSearchCV ----
    param_grid = {
        "modelo__max_depth": [2, 3, 4, 5, None],
        "modelo__min_samples_split": [2, 5, 10, 20],
        "modelo__min_samples_leaf": [1, 2, 5, 10]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=modelo_base,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        refit=True
    )

    grid_search.fit(X_train, y_train)

    modelo = grid_search.best_estimator_

    # ---- Avaliação no teste ----
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    relatorio = classification_report(y_test, y_pred, zero_division=0)

    # ---- Matriz de confusão ----
    arvore = modelo.named_steps["modelo"]
    classes = arvore.classes_
    matriz = confusion_matrix(y_test, y_pred, labels=classes)

    fig_matriz, ax_matriz = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=classes)
    disp.plot(ax=ax_matriz, cmap="Purples", colorbar=False)
    ax_matriz.set_title("Matriz de Confusão - Situação dos Alunos", fontsize=14)
    fig_matriz.tight_layout()

    # ---- Validação cruzada ----
    scores_cv = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    # ---- Importância das features ----
    importancias = arvore.feature_importances_
    df_importancias = pd.DataFrame({
        "Feature": features_numericas,
        "Importancia": importancias
    }).sort_values("Importancia", ascending=False)

    fig_importancias, ax_imp = plt.subplots(figsize=(8, 5))
    ax_imp.barh(df_importancias["Feature"], df_importancias["Importancia"], color="#7c3aed")
    ax_imp.set_xlabel("Importância")
    ax_imp.set_ylabel("Feature")
    ax_imp.set_title("Importância das Features")
    ax_imp.invert_yaxis()
    fig_importancias.tight_layout()

    # ---- Visualização da árvore ----
    fig_arvore, ax_arvore = plt.subplots(figsize=(22, 12))
    plot_tree(
        arvore,
        feature_names=features_numericas,
        class_names=arvore.classes_,
        filled=True,
        rounded=True,
        proportion=False,
        precision=2,
        fontsize=10,
        ax=ax_arvore,
    )
    ax_arvore.set_title("Árvore de Decisão - Classificação da Situação dos Alunos", fontsize=18)
    fig_arvore.tight_layout()

    return {
        "df": df,
        "modelo": modelo,
        "melhores_params": grid_search.best_params_,
        "melhor_score_cv": grid_search.best_score_,
        "acuracia_teste": acuracia,
        "relatorio": relatorio,
        "fig_matriz": fig_matriz,
        "scores_cv": scores_cv,
        "df_importancias": df_importancias,
        "fig_importancias": fig_importancias,
        "fig_arvore": fig_arvore,
    }


resultado_treino = treinar_modelo()
modelo = resultado_treino["modelo"]
arvore = modelo.named_steps["modelo"]


# ============================================================
# 12. FUNÇÃO DE PREDIÇÃO
# ============================================================

def prever_situacao(horas, faltas, nota):
    aluno = pd.DataFrame(
        [[horas, faltas, nota]],
        columns=["Horas_de_estudo", "Faltas", "Nota"]
    )

    previsao = modelo.predict(aluno)[0]
    probabilidades = modelo.predict_proba(aluno)[0]

    probabilidades_formatadas = {
        classe: float(probabilidade)
        for classe, probabilidade in zip(arvore.classes_, probabilidades)
    }

    return previsao, probabilidades_formatadas


def render_resultado_html(previsao, probabilidades_formatadas):
    """
    Renderiza o resultado da previsão como um gráfico de rosca (donut chart)
    feito em CSS puro (conic-gradient), com um selo de confiança central
    e uma legenda lateral — adaptado para o tema escuro (fundo preto).
    """
    cor_principal = CORES_SITUACAO.get(previsao, "#a855f7")

    # Ordena as classes por probabilidade decrescente
    itens_ordenados = sorted(
        probabilidades_formatadas.items(), key=lambda item: item[1], reverse=True
    )
    confianca = itens_ordenados[0][1]

    # Monta os "stops" do conic-gradient (donut chart)
    stops = []
    acumulado = 0.0
    for classe, probabilidade in itens_ordenados:
        cor_classe = CORES_SITUACAO.get(classe, "#a855f7")
        inicio = acumulado * 360
        acumulado += probabilidade
        fim = acumulado * 360
        stops.append(f"{cor_classe} {inicio:.2f}deg {fim:.2f}deg")
    gradiente = ", ".join(stops)

    # Legenda lateral
    legenda_html = ""
    for classe, probabilidade in itens_ordenados:
        cor_classe = CORES_SITUACAO.get(classe, "#a855f7")
        legenda_html += f"""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
            <span style="
                width:10px; height:10px; border-radius:50%;
                background:{cor_classe}; flex-shrink:0;
                box-shadow: 0 0 0 3px {cor_classe}33;
            "></span>
            <span style="font-size:13px; color:#ffffff !important; flex:1;">{classe}</span>
            <span style="font-size:13px; font-weight:700; color:#ffffff !important;">{probabilidade:.1%}</span>
        </div>
        """

    resultado_html = f"""
    <div style="
        background: #0d0d0d;
        border: 1px solid #333333;
        border-radius: 18px;
        padding: 24px 26px;
        font-family: inherit;
    ">
        <div style="font-size:12px; color:#c084fc !important; letter-spacing:0.08em; text-transform:uppercase; font-weight:700; margin-bottom:16px;">
            Diagnóstico do modelo
        </div>

        <div style="display:flex; align-items:center; gap:28px; flex-wrap:wrap;">

            <div style="position:relative; width:150px; height:150px; flex-shrink:0;">
                <div style="
                    width:150px; height:150px; border-radius:50%;
                    background: conic-gradient({gradiente});
                    box-shadow: 0 8px 20px -6px {cor_principal}66;
                "></div>
                <div style="
                    position:absolute; top:50%; left:50%;
                    transform:translate(-50%, -50%);
                    width:98px; height:98px; border-radius:50%;
                    background:#000000;
                    display:flex; flex-direction:column;
                    align-items:center; justify-content:center;
                    box-shadow: inset 0 0 0 1px #444444;
                ">
                    <div style="font-size:22px; font-weight:800; color:{cor_principal} !important; line-height:1;">
                        {confianca:.0%}
                    </div>
                    <div style="font-size:10px; color:#ffffff !important; text-transform:uppercase; letter-spacing:0.05em; margin-top:4px;">
                        confiança
                    </div>
                </div>
            </div>

            <div style="flex:1; min-width:180px;">
                <div style="
                    display:inline-block;
                    padding:6px 16px;
                    border-radius:999px;
                    background:{cor_principal};
                    color:white;
                    font-weight:700;
                    font-size:16px;
                    margin-bottom:14px;
                ">
                    {previsao}
                </div>
                {legenda_html}
            </div>

        </div>
    </div>
    """
    return resultado_html


PLACEHOLDER_HTML = """
<div style="
    border: 1px dashed #444444;
    border-radius: 16px;
    padding: 30px;
    text-align:center;
    color:#ffffff !important;
">
    Preencha os dados e clique em <b>Prever situação</b>.
</div>
"""


# ============================================================
# 13. INTERFACE STREAMLIT — TEMA ESCURO
# ============================================================

st.markdown(
    """
    <div id="cabecalho">
        <h1>🎓 Previsão da Situação do Aluno</h1>
        <p>Preencha os dados abaixo para estimar se o aluno será <b>Aprovado</b>,
        ficará em <b>Recuperação</b> ou será <b>Reprovado</b> — usando um modelo
        de Árvore de Decisão.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_form, col_resultado = st.columns(2, gap="large")

with col_form:
    st.markdown("### 📋 Dados do aluno")

    horas_input = st.number_input(
        "⏱️ Horas de estudo por semana",
        min_value=0.0, max_value=24.0, value=6.0, step=0.5,
    )
    faltas_input = st.number_input(
        "📅 Número de faltas",
        min_value=0, max_value=30, value=2, step=1,
    )
    nota_input = st.number_input(
        "📝 Nota",
        min_value=0.0, max_value=10.0, value=7.0, step=0.1,
    )

    botao = st.button("🔮 Prever situação", type="primary", use_container_width=True)

with col_resultado:
    st.markdown("### 📊 Resultado")
    resultado_placeholder = st.empty()

    horas_calc, faltas_calc, nota_calc = horas_input, faltas_input, nota_input

    if botao:
        try:
            horas_val = float(horas_calc)
            faltas_val = float(faltas_calc)
            nota_val = float(nota_calc)

            if horas_val < 0 or horas_val > 24:
                st.error("Horas de estudo deve estar entre 0 e 24 horas.")
            elif faltas_val < 0 or faltas_val > 30:
                st.error("Faltas deve estar entre 0 e 30.")
            elif nota_val < 0 or nota_val > 10:
                st.error("A nota deve estar entre 0 e 10.")
            else:
                previsao, probabilidades = prever_situacao(horas_val, faltas_val, nota_val)
                resultado_placeholder.markdown(
                    render_resultado_html(previsao, probabilidades),
                    unsafe_allow_html=True,
                )
        except (TypeError, ValueError):
            st.error("Informe apenas valores numéricos.")
    else:
        resultado_placeholder.markdown(PLACEHOLDER_HTML, unsafe_allow_html=True)

st.markdown(
    """
    <div id="rodape">
    ⚠️ Modelo demonstrativo treinado com dados sintéticos.
    Para decisões acadêmicas reais, utilize dados históricos representativos.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 14. DETALHES TÉCNICOS DO MODELO (equivalente aos prints do script original)
# ============================================================

with st.expander("🔧 Detalhes técnicos do modelo (dados, treino e avaliação)"):
    st.markdown("#### Amostra dos dados")
    st.dataframe(resultado_treino["df"].head())

    st.markdown("#### Distribuição das classes")
    st.dataframe(resultado_treino["df"]["Situacao"].value_counts())

    st.markdown("#### Melhores parâmetros (GridSearchCV)")
    st.json(resultado_treino["melhores_params"])
    st.write(f"Melhor acurácia média da validação: **{resultado_treino['melhor_score_cv']:.4f}**")

    st.markdown("#### Avaliação no conjunto de teste")
    st.write(f"Acurácia: **{resultado_treino['acuracia_teste']:.4f}**")
    st.code(resultado_treino["relatorio"])

    st.markdown("#### Matriz de confusão")
    st.pyplot(resultado_treino["fig_matriz"])

    st.markdown("#### Validação cruzada")
    st.write("Scores:", resultado_treino["scores_cv"])
    st.write(f"Média: {resultado_treino['scores_cv'].mean():.4f}")
    st.write(f"Desvio padrão: {resultado_treino['scores_cv'].std():.4f}")

    st.markdown("#### Importância das features")
    st.dataframe(resultado_treino["df_importancias"])
    st.pyplot(resultado_treino["fig_importancias"])

    st.markdown("#### Árvore de decisão")
    st.pyplot(resultado_treino["fig_arvore"])
