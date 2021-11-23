import numpy as np
import matplotlib.pyplot as plt


def distancia_euclideana(ponto_a, ponto_b):
    """Devolve a distancia euclideana entre o ponto a e o ponto b"""
    return np.sqrt(np.sum(np.square(ponto_a - ponto_b)))


class IsagKmeans:
    
    def __init__(self, K=3, max_iterations = 1000):
        self.K = K 
        self.max_iterations = max_iterations
        
        
    def fit(self, X):
        """Utiliza os pontos X para encontrar os K centroides que melhor se
        adaptam aos dados.
        INPUT:
            X - Matriz de dados
        OUTPUT:
            centroides - Os K centros para a matriz X"""
        
        self.X = X
        self.numero_exemplos, self.numero_atributos = X.shape
        
        # iniciar os centroides
        self.centroides = self._iniciar_centroides()
        
        # optimizar
        for _ in range(self.max_iterations):

            # criar clusters
            self.clusters = self._criar_clusters(self.X)

            # atualizar centroides
            centroides_antigos = self.centroides
            self.centroides = self._atualizar_centroides(self.X)

            # confirmar paragem 
            should_stop = self._confirmar_paragem(self.centroides, centroides_antigos)

            if should_stop:
                break
                    
                    
    def predict(self, X, should_plot = False):
        
        # calcular os clusters para X
        clusters = self._criar_clusters(X)
        
        # devolver label
        
        labels = self._devolver_labels(X, clusters)
        
        # fazer plot
        
        if should_plot:
            self.plot(X, clusters)
            
        return labels
        
        
        
    def _devolver_labels(self, X, clusters):
        labels = np.zeros(self.numero_exemplos)
        
        count = 0
        for cluster in clusters:
            for i in cluster:
                labels[i] = count
            count += 1
        return labels
        
        
        
        return labels
    
    def _iniciar_centroides(self, noise = 0.1):
        centroides = []
        for _ in range(self.K):
            centroide_array = []
            for atributo in range(self.numero_atributos):
                media_atributo = np.mean(self.X[:,atributo])
                centroide_valor_atributo = (media_atributo 
                                            + noise 
                                            * ((np.random.rand() * 2) - 1) 
                                            * media_atributo)
                centroide_array.append(centroide_valor_atributo)
            centroides.append(np.array(centroide_array))
        return centroides
    
    
    def _criar_clusters(self, X):
        
        clusters = [[] for _ in range(self.K)]
        
        for i in range(self.numero_exemplos):
            ponto = X[i,:]
            centroide_mais_perto = self._centroide_mais_perto(ponto)
            clusters[centroide_mais_perto].append(i)
            
        return clusters
    
    def _centroide_mais_perto(self, ponto):
        distancia_aos_centroides = [distancia_euclideana(ponto, centroide) 
                                    for centroide in self.centroides]
        centroide_mais_perto = np.argmin(distancia_aos_centroides)
        return centroide_mais_perto
    
    def _atualizar_centroides(self, X):
        centroides = self._iniciar_centroides(noise = 0.1)
        
        count = 0
        for cluster in self.clusters:
            centro_de_gravidade = np.mean(X[cluster], axis = 0)
            centroides[count] = centro_de_gravidade
            count +=1
            
        return centroides
    
    def _confirmar_paragem(self, centroides, centroides_antigos):
        distancia_entre_centroides = [distancia_euclideana(centroides[i], centroides_antigos[i]) 
                                      for i in range(self.K)]
        should_stop = True if np.sum(distancia_entre_centroides) == 0 else False
        
        return should_stop

    
    def plot(self, X, clusters):
        fig, ax = plt.subplots(figsize=(12,12))
        
        for cluster in clusters:
            points = self.X[cluster].T
            ax.scatter(*points)
        
        for centroide in self.centroides:
            ax.scatter(*centroide, marker="x", color = "red", linewidth=4)
        
        plt.show()
