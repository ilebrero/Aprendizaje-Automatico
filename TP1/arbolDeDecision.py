import numpy as np
import pandas as pd

from collections import Counter

def obtener_nueva_altura_restante(altura_restante):
    if (altura_restante == None):
        return None
    else:
        return altura_restante-1

def construir_arbol(instancias, etiquetas, criterion, altura_restante):
    ganancia, pregunta = encontrar_mejor_atributo_y_corte(instancias, etiquetas, criterion)
   
    if ganancia == 0 or (altura_restante != None and altura_restante <= 0):
   
        return Hoja(etiquetas)
    else: 
   
        altura_restante = obtener_nueva_altura_restante(altura_restante)
        
   
        instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen = partir_segun(pregunta, instancias, etiquetas)
   

   
        sub_arbol_izquierdo = construir_arbol(instancias_cumplen, etiquetas_cumplen, criterion, altura_restante)
        sub_arbol_derecho   = construir_arbol(instancias_no_cumplen, etiquetas_no_cumplen, criterion, altura_restante)
   
        
   
        return Nodo_De_Decision(pregunta, sub_arbol_izquierdo, sub_arbol_derecho)



class Hoja:
    
    def __init__(self, etiquetas):
        self.cuentas = dict(Counter(etiquetas))


class Nodo_De_Decision:
    
     
    def __init__(self, pregunta, sub_arbol_izquierdo, sub_arbol_derecho):
        self.pregunta = pregunta
        self.sub_arbol_izquierdo = sub_arbol_izquierdo
        self.sub_arbol_derecho = sub_arbol_derecho
        
        

class Pregunta:
    def __init__(self, atributo, valor):
        self.atributo = atributo
        self.valor = valor
    
    def cumple(self, instancia):

        return instancia[self.atributo] == self.valor
    
    def __repr__(self):
        return "Es el valor para {} igual a {}?".format(self.atributo, self.valor)

def gini(etiquetas):
    total = len(etiquetas)
    
    if (total == 0):
        return 1
    
    positivas = len([i for i in etiquetas if i == 'Si'])
    negativas = len([i for i in etiquetas if i == 'No'])
    
    impureza = 1 - np.square(positivas / total) - np.square(negativas / total)

    return impureza

def ganancia_gini(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha):
    total = len(instancias)
    
    gini_inicial = gini(etiquetas_rama_izquierda + etiquetas_rama_derecha)
    gini_izq = gini(etiquetas_rama_izquierda)
    gini_der = gini(etiquetas_rama_derecha)
    
    prom_ponderado = ((len(etiquetas_rama_izquierda)/total) * gini_izq) + ((len(etiquetas_rama_derecha)/total) * gini_der)
    return (gini_inicial - prom_ponderado)

def entropia(etiquetas):
    total    = len(etiquetas)
    entropia = 0
    
    if (total == 0):
        return 1
    
    muestras_clases = [
        len([i for i in etiquetas if i == 'Si']) / total,
        len([i for i in etiquetas if i == 'No']) / total
    ]
    
    for clase in muestras_clases:
        if (clase != 0):
            entropia = entropia + (-clase * np.log2(clase))
        
    return entropia
        
def ganancia_entropia(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha):
    total = len(instancias)
    
    entropia_inicial = entropia(etiquetas_rama_izquierda + etiquetas_rama_derecha)
    entropia_izq = entropia(etiquetas_rama_izquierda)
    entropia_der = entropia(etiquetas_rama_derecha)
    
    prom_ponderado = ((len(etiquetas_rama_izquierda)/total) * entropia_izq) + ((len(etiquetas_rama_derecha)/total) * entropia_der)
    return (entropia_inicial - prom_ponderado)
    
def partir_segun(pregunta, instancias, etiquetas):
    
    instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen = [], [], [], []
    
    for i in range(len(instancias)):
        etiqueta  = etiquetas[i]
        instancia = instancias.iloc[i]
        
        if (pregunta.cumple(instancia)):
            instancias_cumplen.append(instancia)
            etiquetas_cumplen.append(etiqueta)
        else:
            instancias_no_cumplen.append(instancia)
            etiquetas_no_cumplen.append(etiqueta)
            
            
    
    return pd.DataFrame(instancias_cumplen), etiquetas_cumplen, pd.DataFrame(instancias_no_cumplen), etiquetas_no_cumplen    

def obtener_ganancia(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha, criterion):
    if (criterion == 'gini'):
        return ganancia_gini(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha)
    elif (criterion == 'entropy'):
        return ganancia_entropia(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha)
    else:
        raise ValueError('bad criterion configured | %f not valid'.format(str(criterion)))

def encontrar_mejor_atributo_y_corte(instancias, etiquetas, criterion):
    max_ganancia   = 0
    mejor_pregunta = None
    
    for columna in instancias.columns:
        for valor in set(instancias[columna]):
            
            pregunta = Pregunta(columna, valor)
            _, etiquetas_rama_izquierda, _, etiquetas_rama_derecha = partir_segun(pregunta, instancias, etiquetas)
   
            ganancia = obtener_ganancia(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha, criterion)
            
            if ganancia > max_ganancia:
                max_ganancia = ganancia
                mejor_pregunta = pregunta
            
    return max_ganancia, mejor_pregunta


def imprimir_arbol(arbol, spacing=""):
    if isinstance(arbol, Hoja):
        print (spacing + "Hoja:", arbol.cuentas)
        return

    print (spacing + str(arbol.pregunta))

    print (spacing + '--> True:')
    imprimir_arbol(arbol.sub_arbol_izquierdo, spacing + "  ")

    print (spacing + '--> False:')
    imprimir_arbol(arbol.sub_arbol_derecho, spacing + "  ")


def predecir(arbol, x_t):
    nodo_actual = arbol
    
    while(type(nodo_actual) != Hoja):
        if (nodo_actual.pregunta.cumple(x_t)):
            nodo_actual = nodo_actual.sub_arbol_izquierdo
        else:
            nodo_actual = nodo_actual.sub_arbol_derecho
    
    return next(iter(nodo_actual.cuentas.keys()))
        
class MiClasificadorArbol():
    def __init__(self, max_depth='3', criterion='gini'):
        self.arbol     = None
        self.columnas  = []
        self.max_depth = max_depth
        self.criterion = criterion
    
    def fit(self, X_train, y_train):
        self.columnas = range(len(X_train[0]))
        self.arbol    = construir_arbol(pd.DataFrame(X_train, columns=self.columnas), y_train, self.criterion, self.max_depth)
        return self
    
    def predict(self, X_test):
        predictions = []
        for x_t in X_test:
            x_t_df = pd.DataFrame([x_t], columns=self.columnas).iloc[0]
            prediction = predecir(self.arbol, x_t_df)
            print(x_t, "prediccion ->", prediction)
            predictions.append(prediction)
        return predictions
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        accuracy = sum(y_i == y_j for (y_i, y_j) in zip(y_pred, y_test)) / len(y_test)
        return accuracy
        
