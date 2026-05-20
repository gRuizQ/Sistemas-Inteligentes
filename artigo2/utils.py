import pandas as pd
import numpy as np
import time
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
    inicio = time.perf_counter()
    modelo.fit(X_train, y_train)
    predicoes = modelo.predict(X_test)
    tempo_total = time.perf_counter() - inicio
    
    rmse = np.sqrt(mean_squared_error(y_test, predicoes))
    print(f"--- RESULTADOS DA REGRESSÃO ({nome_modelo}) ---")
    print(f"RMSE {nome_modelo}: {rmse:.4f}")
    print(f"Tempo total ({nome_modelo}): {tempo_total:.4f} s\n")
    
    return modelo


def treinar_e_avaliar_classificacao(modelo, X_train, y_train, X_test, y_test, nome_modelo="Modelo"):
    """
    Treina um modelo de classificação e exibe Acurácia, F1-Score e Matriz de Confusão.
    Retorna o modelo treinado.
    """
    inicio = time.perf_counter()
    modelo.fit(X_train, y_train)
    predicoes = modelo.predict(X_test)
    tempo_total = time.perf_counter() - inicio

    acc = accuracy_score(y_test, predicoes)
    _, _, f1, _ = precision_recall_fscore_support(y_test, predicoes, average='weighted', zero_division=0)
    
    print(f"--- RESULTADOS DA CLASSIFICAÇÃO ({nome_modelo}) ---")
    print(f"{nome_modelo} -> Acurácia: {acc:.4f} | F1-Score: {f1:.4f}")
    print(f"Matriz de Confusão ({nome_modelo}):\n", confusion_matrix(y_test, predicoes), "\n")
    print(f"Tempo total ({nome_modelo}): {tempo_total:.4f} s\n")
    
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
    