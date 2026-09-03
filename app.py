# ============================================================
# PREVISÃO DA SITUAÇÃO DE ALUNOS
# Árvore de Decisão + Avaliação + GridSearchCV + Gradio (tema roxo)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

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
# 1. DADOS
# ============================================================

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

print("\n========== AMOSTRA DOS DADOS ==========")
print(df.head())
print("\n========== DISTRIBUIÇÃO DAS CLASSES ==========")
print(df["Situacao"].value_counts())


# ============================================================
# 2. FEATURES (X) E TARGET (y)
# ============================================================

X = df[["Horas_de_estudo", "Faltas", "Nota"]]
y = df["Situacao"]


# ============================================================
# 3. DIVISÃO TREINO / TESTE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# ============================================================
# 4. PRÉ-PROCESSAMENTO
# ============================================================

features_numericas = ["Horas_de_estudo", "Faltas", "Nota"]
features_categoricas = []

preprocessador = ColumnTransformer(
    transformers=[
        ("numericas", "passthrough", features_numericas),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), features_categoricas)
    ],
    remainder="drop"
)


# ============================================================
# 5. MODELO BASE
# ============================================================

modelo_base = Pipeline(
    steps=[
        ("preprocessamento", preprocessador),
        ("modelo", DecisionTreeClassifier(random_state=42))
    ]
)


# ============================================================
# 6. GRIDSEARCHCV
# ============================================================

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

print("\n========== OTIMIZANDO O MODELO ==========")
grid_search.fit(X_train, y_train)

print("\nMelhores parâmetros:")
print(grid_search.best_params_)
print(f"\nMelhor acurácia média da validação: {grid_search.best_score_:.4f}")

modelo = grid_search.best_estimator_


# ============================================================
# 7. AVALIAÇÃO NO CONJUNTO DE TESTE
# ============================================================

y_pred = modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

print("\n========== AVALIAÇÃO NO TESTE ==========")
print(f"Acurácia: {acuracia:.4f}")
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, zero_division=0))


# ============================================================
# 8. MATRIZ DE CONFUSÃO
# ============================================================

classes = modelo.named_steps["modelo"].classes_
matriz = confusion_matrix(y_test, y_pred, labels=classes)

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=classes)
disp.plot(ax=ax, cmap="Purples", colorbar=False)
ax.set_title("Matriz de Confusão - Situação dos Alunos", fontsize=14)
plt.tight_layout()
plt.savefig("matriz_confusao.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ============================================================
# 9. VALIDAÇÃO CRUZADA
# ============================================================

scores_cv = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

print("\n========== VALIDAÇÃO CRUZADA ==========")
print("Scores:", scores_cv)
print(f"Média: {scores_cv.mean():.4f}")
print(f"Desvio padrão: {scores_cv.std():.4f}")


# ============================================================
# 10. IMPORTÂNCIA DAS FEATURES
# ============================================================

arvore = modelo.named_steps["modelo"]
importancias = arvore.feature_importances_

df_importancias = pd.DataFrame({
    "Feature": features_numericas,
    "Importancia": importancias
}).sort_values("Importancia", ascending=False)

print("\n========== IMPORTÂNCIA DAS FEATURES ==========")
print(df_importancias)

plt.figure(figsize=(8, 5))
plt.barh(df_importancias["Feature"], df_importancias["Importancia"], color="#7c3aed")
plt.xlabel("Importância")
plt.ylabel("Feature")
plt.title("Importância das Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("importancia_features.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 11. VISUALIZAÇÃO DA ÁRVORE
# ============================================================

plt.figure(figsize=(22, 12))
plot_tree(
    arvore,
    feature_names=features_numericas,
    class_names=arvore.classes_,
    filled=True,
    rounded=True,
    proportion=False,
    precision=2,
    fontsize=10
)
plt.title("Árvore de Decisão - Classificação da Situação dos Alunos", fontsize=18)
plt.tight_layout()
plt.savefig("arvore_decisao.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 12. FUNÇÃO DE PREDIÇÃO
# ============================================================

# Cor associada a cada classe, usada nos "chips" de resultado da interface
CORES_SITUACAO = {
    "Aprovado": "#22c55e",
    "Recuperação": "#f59e0b",
    "Reprovado": "#ef4444",
}


def prever_situacao(horas, faltas, nota):

    if horas is None or faltas is None or nota is None:
        raise gr.Error("Preencha todos os campos.")

    try:
        horas = float(horas)
        faltas = float(faltas)
        nota = float(nota)
    except (TypeError, ValueError):
        raise gr.Error("Informe apenas valores numéricos.")

    if horas < 0 or horas > 24:
        raise gr.Error("Horas de estudo deve estar entre 0 e 24 horas.")
    if faltas < 0 or faltas > 30:
        raise gr.Error("Faltas deve estar entre 0 e 30.")
    if nota < 0 or nota > 10:
        raise gr.Error("A nota deve estar entre 0 e 10.")

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

    cor = CORES_SITUACAO.get(previsao, "#7c3aed")

    # Card de resultado em HTML, com "chip" colorido e barras de probabilidade
    barras_html = ""
    for classe, probabilidade in sorted(
        probabilidades_formatadas.items(), key=lambda item: item[1], reverse=True
    ):
        cor_classe = CORES_SITUACAO.get(classe, "#7c3aed")
        largura = max(probabilidade * 100, 3)
        barras_html += f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; font-size:13px; color:#4c1d95; margin-bottom:4px;">
                <span>{classe}</span>
                <span>{probabilidade:.1%}</span>
            </div>
            <div style="background:#ede9fe; border-radius:8px; height:10px; overflow:hidden;">
                <div style="width:{largura}%; background:{cor_classe}; height:100%; border-radius:8px;"></div>
            </div>
        </div>
        """

    resultado_html = f"""
    <div style="
        background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
        border: 1px solid #ddd6fe;
        border-radius: 16px;
        padding: 20px 22px;
        font-family: inherit;
    ">
        <div style="font-size:13px; color:#7c3aed; letter-spacing:0.05em; text-transform:uppercase; font-weight:600;">
            Situação prevista
        </div>
        <div style="
            display:inline-block;
            margin-top:8px;
            margin-bottom:18px;
            padding:6px 16px;
            border-radius:999px;
            background:{cor};
            color:white;
            font-weight:700;
            font-size:18px;
        ">
            {previsao}
        </div>
        <div style="font-size:13px; color:#6d28d9; font-weight:600; margin-bottom:10px;">
            Probabilidade por classe
        </div>
        {barras_html}
    </div>
    """

    return resultado_html


# ============================================================
# 13. INTERFACE GRADIO — TEMA ROXO
# ============================================================

tema_roxo = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui"],
).set(
    body_background_fill="linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%)",
    block_background_fill="white",
    block_border_color="#e9d5ff",
    block_radius="18px",
    block_shadow="0 8px 24px rgba(124, 58, 237, 0.08)",
    button_primary_background_fill="linear-gradient(90deg, #7c3aed 0%, #a855f7 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #6d28d9 0%, #9333ea 100%)",
    button_primary_text_color="white",
    input_background_fill="#faf5ff",
    input_border_color="#ddd6fe",
    slider_color="#7c3aed",
)

# CSS extra para refinar cabeçalho, cards e detalhes que o theme não cobre
CSS_EXTRA = """
#cabecalho {
    background: linear-gradient(90deg, #7c3aed 0%, #a855f7 60%, #c084fc 100%);
    padding: 28px 32px;
    border-radius: 20px;
    color: white !important;
    margin-bottom: 8px;
}
#cabecalho h1, #cabecalho p {
    color: white !important;
}
#rodape {
    text-align: center;
    color: #8b5cf6;
    font-size: 13px;
    margin-top: 10px;
}
.gradio-container {
    max-width: 980px !important;
    margin: auto !important;
}
"""

with gr.Blocks(theme=tema_roxo, css=CSS_EXTRA, title="Previsão da Situação do Aluno") as interface:

    with gr.Column(elem_id="cabecalho"):
        gr.Markdown(
            """
            # 🎓 Previsão da Situação do Aluno
            Preencha os dados abaixo para estimar se o aluno será **Aprovado**,
            ficará em **Recuperação** ou será **Reprovado** — usando um modelo
            de Árvore de Decisão.
            """
        )

    with gr.Row(equal_height=True):

        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 📋 Dados do aluno")

            horas_input = gr.Number(
                label="⏱️ Horas de estudo por semana",
                minimum=0, maximum=24, value=6, step=0.5,
            )
            faltas_input = gr.Number(
                label="📅 Número de faltas",
                minimum=0, maximum=30, value=2, step=1,
            )
            nota_input = gr.Number(
                label="📝 Nota",
                minimum=0, maximum=10, value=7, step=0.1,
            )

            botao = gr.Button("🔮 Prever situação", variant="primary", size="lg")

            gr.Examples(
                examples=[
                    [10, 1, 9.0],
                    [6, 3, 7.0],
                    [4, 8, 5.5],
                    [2, 15, 3.5],
                ],
                inputs=[horas_input, faltas_input, nota_input],
                label="✨ Exemplos rápidos",
            )

        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 📊 Resultado")
            resultado_output = gr.HTML(
                value="""
                <div style="
                    border: 1px dashed #ddd6fe;
                    border-radius: 16px;
                    padding: 30px;
                    text-align:center;
                    color:#8b5cf6;
                ">
                    Preencha os dados e clique em <b>Prever situação</b>.
                </div>
                """
            )

    gr.Markdown(
        """
        <div id="rodape">
        ⚠️ Modelo demonstrativo treinado com dados sintéticos.
        Para decisões acadêmicas reais, utilize dados históricos representativos.
        </div>
        """
    )

    botao.click(
        fn=prever_situacao,
        inputs=[horas_input, faltas_input, nota_input],
        outputs=resultado_output,
    )


# ============================================================
# 14. EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    print("\n========================================")
    print("Aplicação iniciada.")
    print("Arquivos gerados:")
    print(" - arvore_decisao.png")
    print(" - matriz_confusao.png")
    print(" - importancia_features.png")
    print("========================================\n")

    interface.launch()
