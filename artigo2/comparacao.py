import os

import utils
import algoritmo_cart as cart
import random_forest as rf
from sklearn.neural_network import MLPRegressor, MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

# Silencia especificamente o aviso de "não convergência" do lbfgs
warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = os.path.dirname(__file__)

# 1. Preparação dos dados
arquivo_treino = os.path.join(BASE_DIR, '02_treino_sinais_vitais_com_label.txt')
X_train_s, X_test_s, y_reg_train, y_reg_test, y_clf_train, y_clf_test, scaler = utils.preparar_dados(arquivo_treino)

# 2. Definindo os dicionários de modelos para testar (Diferentes algoritmos e parâmetros)
modelos_para_regressao = {
    "MLP_9_9_7_sigmoid": MLPRegressor(hidden_layer_sizes=(9, 9, 7), tol=0.001, solver='lbfgs', max_iter=2000, learning_rate_init=0.01, random_state=42, activation="logistic"),
    "CART_Custom": cart.CARTDecisionTreeRegressor(random_state=42),
    "RandomForest_Custom": rf.RandomForestRegressorCustom(
        n_estimators=300, random_state=42
    ),
}

modelos_para_classificacao = {
    "MLP_8_6_tanh": MLPClassifier(hidden_layer_sizes=(8, 5), tol=0.001, solver='lbfgs', max_iter=2000, learning_rate_init=0.01, random_state=42, activation="tanh"),
    "CART_Custom": cart.CARTDecisionTreeClassifier(random_state=42),
    "RandomForest_Custom": rf.RandomForestClassifierCustom(
        n_estimators=300, random_state=42
    ),
}

# 3. Treino e Avaliação em Lote
resultados_regressao = utils.comparar_modelos(
    modelos_para_regressao, X_train_s, y_reg_train, X_test_s, y_reg_test, tarefa='regressao'
)

resultados_classificacao = utils.comparar_modelos(
    modelos_para_classificacao, X_train_s, y_clf_train, X_test_s, y_clf_test, tarefa='classificacao'
)

# 4. Geração do teste cego (Selecionando o melhor modelo manualmente após ver o console)
arquivo_cego = os.path.join(BASE_DIR, '01_treino_sinais_vitais_sem_label.txt')
arquivo_saida = os.path.join(BASE_DIR, 'resultados_predicao.csv')

# Suponha que ao analisar o console, a MLP (10, 5, 5) foi a melhor na regressão 
# e a Random Forest foi a melhor na classificação:
melhor_modelo_reg = resultados_regressao["MLP_9_9_7_sigmoid"]
melhor_modelo_clf = resultados_classificacao["MLP_8_6_tanh"]

utils.gerar_arquivo_teste_cego(
    filepath_entrada=arquivo_cego,
    filepath_saida=arquivo_saida,
    scaler=scaler,
    modelo_reg=melhor_modelo_reg,
    modelo_clf=melhor_modelo_clf
)