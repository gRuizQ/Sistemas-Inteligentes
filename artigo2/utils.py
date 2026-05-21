import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_recall_fscore_support, confusion_matrix

def preparar_dados(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Carrega o dataset, separa features/targets, divide em treino/teste e aplica o escalonamento.
    """
    colunas = ['i', 'si1', 'si2', 'si3', 'si4', 'si5', 'gi', 'yi']
    df = pd.read_csv(filepath, sep=',', header=None, names=colunas)

    # Separando as Features (X) e os Targets (y). 
    X = df[['si3', 'si4', 'si5']] # qPA, pulso, respiração
    y_reg = df['gi']              # Target da Regressão (Gravidade)
    y_clf = df['yi']              # Target da Classificação (Classe)

    # Dividindo em Treinamento e Validação/Teste
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=test_size, random_state=random_state
    )

    # Escalonamento 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_reg_train, y_reg_test, y_clf_train, y_clf_test, scaler


def treinar_e_avaliar_regressao(modelo, X_train, y_train, X_test, y_test, nome_modelo="Modelo"):
    """
    Treina um modelo de regressão e avalia usando RMSE.
    Retorna o modelo treinado.
    """
    modelo.fit(X_train, y_train)
    predicoes = modelo.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predicoes))
    print(f"--- RESULTADOS DA REGRESSÃO ({nome_modelo}) ---")
    print(f"RMSE {nome_modelo}: {rmse:.4f}")
    print()
    
    return modelo


def treinar_e_avaliar_classificacao(modelo, X_train, y_train, X_test, y_test, nome_modelo="Modelo"):
    """
    Treina um modelo de classificação e exibe Acurácia, F1-Score e Matriz de Confusão.
    Retorna o modelo treinado.
    """
    modelo.fit(X_train, y_train)
    predicoes = modelo.predict(X_test)

    acc = accuracy_score(y_test, predicoes)
    _, _, f1, _ = precision_recall_fscore_support(y_test, predicoes, average='weighted', zero_division=0)
    
    print(f"--- RESULTADOS DA CLASSIFICAÇÃO ({nome_modelo}) ---")
    print(f"{nome_modelo} -> Acurácia: {acc:.4f} | F1-Score: {f1:.4f}")
    print(f"Matriz de Confusão ({nome_modelo}):\n", confusion_matrix(y_test, predicoes), "\n")
    
    return modelo

def comparar_modelos(dicionario_modelos: dict, X_train, y_train, X_test, y_test, tarefa: str):
    """
    Itera sobre um dicionário de modelos, executa o treinamento/avaliação 
    apropriado e retorna um dicionário com os modelos já treinados.
    
    tarefa: deve ser 'regressao' ou 'classificacao'
    """
    print(f"\n{'='*50}")
    print(f" INICIANDO COMPARAÇÃO DE MODELOS: {tarefa.upper()}")
    print(f"{'='*50}\n")
    
    modelos_treinados = {}
    
    for nome_modelo, modelo_instanciado in dicionario_modelos.items():
        if tarefa == 'regressao':
            modelo_treinado = treinar_e_avaliar_regressao(
                modelo_instanciado, X_train, y_train, X_test, y_test, nome_modelo=nome_modelo
            )
        elif tarefa == 'classificacao':
            modelo_treinado = treinar_e_avaliar_classificacao(
                modelo_instanciado, X_train, y_train, X_test, y_test, nome_modelo=nome_modelo
            )
        else:
            raise ValueError("A tarefa deve ser 'regressao' ou 'classificacao'.")
            
        modelos_treinados[nome_modelo] = modelo_treinado
        
    return modelos_treinados


def gerar_arquivo_teste_cego(filepath_entrada: str, filepath_saida: str, scaler, modelo_reg, modelo_clf):
    """
    Carrega o dataset cego, aplica o escalonador treinado, realiza as predições e exporta para CSV.
    """
    colunas = ['i', 'si1', 'si2', 'si3', 'si4', 'si5', 'gi']
    df_cego = pd.read_csv(filepath_entrada, sep=',', header=None, names=colunas)
    
    X_cego = df_cego[['si3', 'si4', 'si5']]
    X_cego_scaled = scaler.transform(X_cego) 

    # Prevendo gravidade e classe
    previsoes_gravidade = modelo_reg.predict(X_cego_scaled)
    previsoes_classe = modelo_clf.predict(X_cego_scaled)

    # Montando o DataFrame de saída
    resultado_final = pd.DataFrame({
        'i': df_cego['i'],
        'gravid': previsoes_gravidade,
        'classe': previsoes_classe
    })

    # Exportando para CSV no formato exigido
    resultado_final.to_csv(filepath_saida, index=False)
    print(f"Arquivo '{filepath_saida}' gerado com sucesso!")
    
def plotar_analise_regressao(y_true, y_pred, nome_modelo="Modelo Regressão"):
    """
    Gera gráficos de diagnóstico para modelos de regressão:
    1. Gráfico de Dispersão (Reais vs. Preditos)
    2. Gráfico de Erros
    """
    # Configurando o tamanho da figura para ter dois gráficos lado a lado
    fig, eixos = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Análise de Desempenho - {nome_modelo}', fontsize=14, fontweight='bold')

    # --- Gráfico 1: Valores Reais vs Valores Preditos ---
    # O ideal é que os pontos fiquem o mais próximo possível da linha diagonal vermelha.
    eixos[0].scatter(y_true, y_pred, alpha=0.6, color='dodgerblue', edgecolors='k')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    eixos[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Previsão')
    
    eixos[0].set_xlabel("Valores Reais (Gravidade)")
    eixos[0].set_ylabel("Valores Preditos")
    eixos[0].set_title("Reais vs Preditos")
    eixos[0].legend()
    eixos[0].grid(True, linestyle='--', alpha=0.7)

    # --- Gráfico 2: Distribuição dos Erros ---
    residuos = y_true - y_pred
    eixos[1].scatter(y_pred, residuos, alpha=0.6, color='tomato', edgecolors='k')
    eixos[1].axhline(y=0, color='black', linestyle='-', lw=2, label='Erro Zero')
    
    eixos[1].set_xlabel("Valores Preditos")
    eixos[1].set_ylabel("Erros (Real - Predito)")
    eixos[1].set_title("Análise de Erros")
    eixos[1].legend()
    eixos[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def plotar_analise_classificacao(y_true, y_pred, nome_modelo="Modelo Classificação"):
    """
    Gera um Heatmap (Mapa de Calor) da Matriz de Confusão para avaliar 
    os acertos e erros de cada classe específica.
    """
    plt.figure(figsize=(7, 5))
    
    # Calcula a matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    # Plota usando o Seaborn para um visual mais limpo
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5, cbar=True)
    
    plt.title(f'Matriz de Confusão - {nome_modelo}', fontsize=14, fontweight='bold')
    plt.xlabel("Classe Predita (y'i)")
    plt.ylabel('Classe Real (yi)')
    
    plt.tight_layout()
    plt.show()
