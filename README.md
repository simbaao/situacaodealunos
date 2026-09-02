# 🎓 Previsão da Situação do Aluno

Modelo de Machine Learning que prevê se um aluno será **Aprovado**, ficará em **Recuperação** ou será **Reprovado**, com base em três variáveis:

- Horas de estudo semanais
- Número de faltas
- Nota

O projeto inclui geração de dados sintéticos, treinamento de uma Árvore de Decisão com otimização de hiperparâmetros, avaliação completa do modelo e uma interface interativa feita com [Gradio](https://www.gradio.app/), em tema roxo.

---

## 📸 Visão geral

A interface permite inserir os dados de um aluno e receber:

- A situação prevista, destacada como um "chip" colorido (verde/laranja/vermelho)
- As probabilidades de cada classe, exibidas em barras
- Exemplos prontos para testar rapidamente

---

## 🗂️ Estrutura do projeto

```
.
├── modelo_aprovacao_aluno_roxo.py   # Script principal (treino + interface Gradio)
├── arvore_decisao.png               # Gerado ao rodar o script
├── matriz_confusao.png              # Gerado ao rodar o script
├── importancia_features.png         # Gerado ao rodar o script
└── README.md
```

---

## ⚙️ Como funciona

O script executa, em ordem, as seguintes etapas:

1. **Geração dos dados**
   Cria um dataset sintético com 500 registros, simulando a relação entre horas de estudo, faltas e nota, com ruído aleatório para maior realismo.
   > ⚠️ Para uso real, substitua essa etapa por dados históricos reais da instituição (ex.: `pd.read_csv("seus_dados.csv")`).

2. **Pré-processamento**
   Separa features numéricas e categóricas usando `ColumnTransformer` (hoje só há features numéricas, mas a estrutura já suporta variáveis categóricas futuras, como turno ou curso).

3. **Treinamento e otimização**
   Usa `GridSearchCV` com `StratifiedKFold` para encontrar os melhores hiperparâmetros da Árvore de Decisão (`max_depth`, `min_samples_split`, `min_samples_leaf`), evitando overfitting.

4. **Avaliação do modelo**
   Calcula acurácia, relatório de classificação (`classification_report`), matriz de confusão e validação cruzada (`cross_val_score`).

5. **Visualizações**
   Gera e salva três imagens:
   - `arvore_decisao.png` — estrutura da árvore treinada
   - `matriz_confusao.png` — desempenho do modelo no conjunto de teste
   - `importancia_features.png` — peso de cada variável na decisão do modelo

6. **Interface Gradio**
   Interface web local para inserir os dados de um aluno e visualizar a previsão com probabilidades, em um layout roxo customizado.

---

## 🚀 Como rodar

### 1. Pré-requisitos

- Python 3.9 ou superior

### 2. Instalar as dependências

```bash
pip install pandas numpy scikit-learn matplotlib gradio
```

### 3. Executar

```bash
python modelo_aprovacao_aluno_roxo.py
```

O terminal vai mostrar:
- Amostra dos dados e distribuição das classes
- Melhores hiperparâmetros encontrados
- Métricas de avaliação (acurácia, relatório de classificação, validação cruzada)
- Importância das features

Em seguida, a interface Gradio abre automaticamente no navegador (geralmente em `http://127.0.0.1:7860`).

---

## 🖱️ Usando a interface

1. Informe **horas de estudo**, **número de faltas** e **nota** do aluno.
2. Clique em **🔮 Prever situação**.
3. Veja o resultado destacado e as probabilidades de cada classe.

Também é possível clicar em um dos **exemplos rápidos** para preencher os campos automaticamente.

### Limites de entrada

| Campo             | Mínimo | Máximo |
|-------------------|--------|--------|
| Horas de estudo   | 0      | 24     |
| Faltas            | 0      | 30     |
| Nota              | 0      | 10     |

Valores fora desses limites ou não numéricos geram uma mensagem de erro na própria interface.

---

## 🌲 Sobre o modelo

- **Algoritmo:** Árvore de Decisão (`DecisionTreeClassifier`)
- **Por que árvore de decisão?** É um modelo facilmente interpretável — importante em contexto educacional, onde é útil entender *por que* o modelo chegou a determinada previsão.
- **Ajuste de hiperparâmetros:** feito via `GridSearchCV`, testando profundidade máxima e critérios mínimos de divisão/folha, para reduzir overfitting.
- **Validação:** validação cruzada estratificada (5 folds) garante que o desempenho seja consistente entre diferentes divisões dos dados.

---

## ⚠️ Aviso importante

Este projeto usa **dados sintéticos** gerados artificialmente, apenas para fins demonstrativos e educacionais. O modelo **não deve ser usado para decisões acadêmicas reais** sem antes ser retreinado com dados históricos reais e validado pela instituição de ensino responsável.

---

## 🔧 Possíveis melhorias futuras

- Substituir os dados sintéticos por uma base real (CSV ou banco de dados)
- Adicionar novas features (ex.: turno, curso, participação em atividades)
- Comparar a Árvore de Decisão com outros modelos (Random Forest, Regressão Logística)
- Persistir o modelo treinado em disco (`joblib`) para não precisar retreinar a cada execução
- Publicar a interface Gradio (`interface.launch(share=True)` ou deploy no Hugging Face Spaces)

---

## 📄 Licença

Defina aqui a licença do seu projeto (ex.: MIT, Apache 2.0) ou remova esta seção se for um projeto interno/acadêmico.
