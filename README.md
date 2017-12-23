# Retificacoes-Geometria-Projetiva
Projeto de Implementação de Retificações para a cadeira de Mídias

## Execução ##
Escolha a imagem na qual serão feitas as retificações, em seguida escolha uma das 5 opções abaixo:

Instruções das retificações:

* A escolha dos pontos é feita com duplo clique em um ponto da imagem. Para sair da seleção de pontos, aperte _Esc_.

1. __Retificacao com 4 pontos:__ Escolha 4 pontos que formarão um quadrado na imagem retificada, a transformação os transformará em pontos de um quadrado.
2. __Retificacao com a reta do infinito:__ Insira os parâmetros de uma reta (x, y, z) que você quer levar para o infinito, separando-os por quebra de linha.
3. __Retificacao com dois pares de retas paralelas:__ Escolha 4 pontos: X1, X2, X3 e X4, o programa formará 4 retas: (X1×X2), (X3×X4), (X1×X3) e (X2×X4) e tornará as duas primeiras e as duas últimas paralelas entre si.
4. __Retificacao com dois pares de retas ortogonais:__ Escolha 2 trios de pontos. Cada trio de pontos X1, X2, X3 formará duas retas (X1×X2) e (X1×X3) que serão transformadas em retas ortogonais. Esta retificação só funciona em imagens que já passaram por uma retificação afim (Xadrez3.png).
5. __Retificacao com cinco pares de retas ortogonais:__ Similar ao 4, porém com 5 pares de retas. Funciona em qualquer caso.

## Imports ##

* __numpy:__ Foram usadas funções para produtos (vetorial e escalar), copiar matrizes, transposta, inversa, calculo de determinante e SVD.
* __cv2 (opencv):__ Foram usadas funções para leitura, divisão e junção do RGB da imagem (para a transposta da imagem, já que usar a função ```numpy.transpose()``` direto também mudaria a dimensão das cores). As demais funções são usadas na interface para selecionar os pontos. ```EVENT_LBUTTONDBLCLK``` ```setMouseCallback()``` e ```waitKey()``` administram entradas do mouse e teclado, ```line()``` e ```circle()``` desenham as retas e pontos selecionados.
* __math:__ A função ```sqrt()```, usada nas transformações 4 e 5 para obter a transformação.
* __tkinter:__ Janela de seleção de imagens.

## Algoritmo ##

### Preparo da imagem ###

O programa inicialmente obtem a transposta da imagem e cria uma "moldura" (quadrado de pixels pretos ao redor de imagem) para que ela fique adequada ao problema (já que as coordenadas no opencv são (y,x)) e pedaços dos pixels na borda da imagem não escapem da janela após a transformação.

### Geração das transformações ###

Após isso, ele gerará a transformação ``t`` escolhida dentre as 5 acima. O algoritmo varia para cada opção:

* As equações lineares em todos os casos abaixo foram resolvidas com a regra de Cramer.

1. __Retificacao com 4 pontos:__ Aplica o DLT nos pontos de forma a gerar uma transformação que os transforme em um quadrado, gerando um sistema de equações lineares.
2. __Retificacao com a reta do infinito:__ Produz a matriz que transforma a reta na reta do infinito (livro do Hartley, eq. 2.19, pág. 49)
3. __Retificacao com dois pares de retas paralelas:__ Realiza o produto vetorial nos dois pares de retas paralelas, gerando dois pontos no infinito. Esses pontos geram a reta que será transformada na reta no infinito.
4. __Retificacao com dois pares de retas ortogonais:__ Utiliza uma fórmula para obter C\*∞' (livro do Hartley, ex. 2.26, pág. 56), gerando um sistema de equações lineares. Em seguida obtem a transformação realizando o SVD, mutiplicando U pela raiz de D (que é uma transformação mD tal que mD\*mD = D, como D é uma mudança de escala, cada valor em alguma posição de mD é a raiz do valor de D na mesma posição).
5. __Retificacao com cinco pares de retas ortogonais:__ Nesse caso usa-se outra equação apra gerar o sistema de equações (livro do Hartley, ex. 2.27, pág. 57), a partir daí, o processo é o mesmo do caso 4.

### Bounding Box ###

Realiza se a transformação nos quatro extremos da imagem, para obter os valores mínimo e máximo de x e y.
Em seguida é calculada uma escala da imagem para que ela encaixe exatamente em uma largura e altura máximas da janela da imagem.
A escala e os valores mínimos de x e y são usados para gerar uma transformação ``t2`` que altera a escala e realiza uma translação da imagem para os limites do bounding box coincidirem com os limites da janela.

### Produção da imagem ###

A imagem passa pela transformação ``dot(t2, t)`` que retifica e encaixa a imagem na janela. Para transformar a imagem, cada ponto na nova janela passa pela transformação inversa para chegar no ponto original, evitando buracos de pontos aos quais nenhum ponto da imagem original chegaria.
