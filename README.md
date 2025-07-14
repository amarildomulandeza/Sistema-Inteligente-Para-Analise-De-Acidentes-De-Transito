
# Sistema Inteligente Para Analise De Acidentes De Transito

**Última Atualização: *14 de Julho de 2025***

1. Demonstração  
2. O que é o Sistema de Deteção de Acidentes?  
3. Pré-requisitos  
4. Como começar - Como utilizar?  
5. Descrição  
6. Trabalhos Futuros  

## 1. Demonstração

![Demonstração](https://user-images.githubusercontent.com/54409969/173066273-732f7da9-8645-4809-aa7a-bb2f78548b3e.gif)

## 2. O que é o Sistema de Deteção de Acidentes?

Um Sistema de Deteção de Acidentes é desenvolvido para detetar acidentes através de vídeos ou imagens de câmaras de vigilância (CCTV). Os acidentes rodoviários são um problema significativo a nível mundial, causando a perda de muitas vidas. Este problema pode ser minimizado utilizando sistemas de deteção de acidentes através de CCTV. Este repositório explora sobretudo como as câmaras de vigilância podem detetar acidentes com a ajuda de técnicas de Aprendizagem Profunda (Deep Learning).

## 3. Pré-requisitos

- Para utilizar este projeto é recomendada a versão do Python > 3.6.  
- Para contribuir para este projeto, é útil ter conhecimentos básicos de scripting em Python, Aprendizagem Automática (Machine Learning) e Aprendizagem Profunda (Deep Learning).

## 4. Como começar - Como utilizar?

### Clonar este repositório

`https://github.com/amarildomulandeza/Sistema-Inteligente-Para-Analise-De-Acidentes-De-Transito.git`

Para instalar todos os pacotes necessários para executar este programa em Python:  
`pip install -r requirements.txt`

**Nota:** Este projeto requer uma câmara. Certifique-se de que tem uma câmara ligada ao seu dispositivo. Também pode utilizar um vídeo previamente descarregado, caso não queira utilizar uma câmara.

### Executar
Antes de executar o programa, é necessário correr o ficheiro `accident-classification.ipynb`, que irá criar o ficheiro `model_weights.h5`. Depois, para correr o programa em Python, deve executar o ficheiro `main.py`.

## 5. Descrição

Este programa inclui 4 componentes:

1. `data`: Conjunto de dados do Kaggle sobre [Deteção de Acidentes a partir de imagens de CCTV](https://www.kaggle.com/code/mrcruise/accident-classification/data).
2. `accident-classification.ipynb`: Notebook Jupyter que gera um modelo para classificar os dados referidos. Este ficheiro gera dois ficheiros importantes: `model.json` e `model_weights.h5`.
3. `detection.py`: Este ficheiro carrega o sistema de deteção de acidentes com a ajuda dos ficheiros `model.json` e `model_weights.h5`.
4. `camera.py`: Este ficheiro gere a câmara e executa o `detection.py` sobre o vídeo, dividindo-o frame a frame e apresentando a percentagem da previsão da presença de acidente em cada frame (se existente).

## 6. Trabalhos Futuros

Pode-se integrar um sistema de alarme que ligue automaticamente para a esquadra de polícia mais próxima em caso de acidente, bem como informe sobre a gravidade do mesmo.
