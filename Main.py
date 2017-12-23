#array e zeros: Criação de arrays
#cross: Produto vetorial
#copy: Copia de vetores
#dot: Multiplicacao de matrizes
#transpose: Matriz transposta
#uint8 e float32: Tipos numericos
from numpy import array, zeros, cross, copy, dot, transpose, uint8, float32
#inv: Matriz inversa
#det: Determinante
#svd: Algoritmo SVD
from numpy.linalg import inv, det, svd
#imread: Leitura da imagem
#imwrite: Salva a imagem
#split: Divisao das cores da imagem
#merge: Juncao das cores da imagem
#EVENT_LBUTTONDBLCLK: Duplo clique com botao esquerdo do mouse
#line e circle: Funcoes para desenhar na imagem
#namedWindow e setMouseCallback: Adicionam a funcao para selecionar os pontos da tela
#imshow: Exibe a imagem
#waitKey: Recebe entradas do teclado
#destroyAllWindows: Fecha todas as janelas
from cv2 import imread, imwrite, split, merge, EVENT_LBUTTONDBLCLK, line, circle, namedWindow, setMouseCallback, imshow, waitKey, destroyAllWindows
from math import sqrt
#Leitura de arquivos
from tkinter.filedialog import askopenfilename

def ajustar_imagem(img):
	'''
	Adiciona uma moldura na imagem para ela nao sair do boundBox e reordena as coordenadas da imagem.
	'''
	(lin, col, _) = img.shape
	moldura = zeros((lin+2, col+2, 3), dtype=uint8)
	moldura[1:-1, 1:-1, :] = img[:, :, :]
	b,g,r = split(moldura)
	b = transpose(b)
	g = transpose(g)
	r = transpose(r)
	return merge((b,g,r))

def transposta(img):
	'''
	Similar ao ajustar_imagem, porem sem a moldura. Usado para visualizacao.
	'''
	b,g,r = split(img)
	b = transpose(b)
	g = transpose(g)
	r = transpose(r)
	return merge((b,g,r))
	
def ret_afim_4_pontos(ponto1, ponto2, ponto3, ponto4):
	'''
	Produz uma retificacao que transforma 4 pontos em vertices de um quadrado.
	'''
	#novos pontos
	#x1' = (0, 0)	x2' = (1, 0)	x3' = (1, 1)	x4' = (0, 1)
	a = array([[0, 0, 0, -ponto1[0], -ponto1[1], -1, 0, 0, 0],
				[ponto1[0], ponto1[1], 1, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, -ponto2[0], -ponto2[1], -1, 0, 0, 0],
				[ponto2[0], ponto2[1], 1, 0, 0, 0, -ponto2[0], -ponto2[1], -1],
				[0, 0, 0, -ponto3[0], -ponto3[1], -1, ponto3[0], ponto3[1], 1],
				[ponto3[0], ponto3[1], 1, 0, 0, 0, -ponto3[0], -ponto3[1], -1],
				[0, 0, 0, -ponto4[0], -ponto4[1], -1, ponto4[0], ponto4[1], 1],
				[ponto4[0], ponto4[1], 1, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 0, 0, 0, 1]])
	detA = det(a)
	t = zeros((3, 3))
	res = array([0, 0, 0, 0, 0, 0, 0, 0, 1])
	for i in range(9):
		aI = copy(a)
		aI[:,i] = res
		# Calculando os valores de t usando DLT com regra de Cramer
		t[i//3,i%3] = det(aI) / detA
		# print("Determinante", i, "=", det(aI))
	print("Retificacao:")
	print(t)
	return t

def ret_afim_reta_inf(reta):
	'''
	Produz uma retificacao que leva uma reta ao infinito.
	'''
	t = zeros((3, 3))
	t[0,0] = 1
	t[1,1] = 1
	t[2] = copy(reta)
	print("Retificacao:")
	print(t)
	return t

def ret_retas_ortogonais(linL1, linM1, linL2, linM2):
	'''
	Produz uma transformacao que torna dois pares de retas ortogonais. Usado apenas em imagens que já passaram por uma retificacao afim.
	'''
	linL1 = normalizar(linL1)
	linM1 = normalizar(linM1)
	linL2 = normalizar(linL2)
	linM2 = normalizar(linM2)
	m = array(
		[[linL1[0]*linM1[0], linL1[0]*linM1[1] + linL1[1]*linM1[0], linL1[1]*linM1[1]],
		[linL2[0]*linM2[0], linL2[0]*linM2[1] + linL2[1]*linM2[0], linL2[1]*linM2[1]],
		[0, 0, 1]])
	detM = det(m)
	s = zeros(3)
	res = array([0, 0, 1])
	for i in range(3):
		mI = copy(m)
		mI[:,i] = res
		s[i] = det(mI) / detM
	conica = array([[s[0], s[1], 0],
						[s[1], s[2], 0],
						[0, 0, 0]])
	print("Conica:")
	print(conica)
	(u, d, _) = svd(conica)
	print("Transformacao:")
	mD = array([[sqrt(d[0]), 0, 0], [0, sqrt(d[1]), 0], [0, 0, 1]])
	saida = dot(u, mD)
	print(saida)
	print("Conica reconstruida:")
	temp = zeros((3,3))
	temp[0,0] = 1
	temp[1,1] = 1
	print(dot(dot(saida, temp), transpose(saida)))	#Checando se a transformacao esta de acordo com a equacao do livro
	return inv(saida)

def ret_retas_ortogonais_2(retas):
	'''
	Produz uma transformacao que torna 5 pares de retas ortogonais.
	'''
	#(l1m1, (l1m2 + l2m1)/2, l2m2, (l1m3 + l3m1)/2, (l2m3 + l3m2)/2, l3m3) c = 0
	print(retas)
	linL1 = normalizar(retas[0])
	linM1 = normalizar(retas[1])
	linL2 = normalizar(retas[2])
	linM2 = normalizar(retas[3])
	linL3 = normalizar(retas[4])
	linM3 = normalizar(retas[5])
	linL4 = normalizar(retas[6])
	linM4 = normalizar(retas[7])
	linL5 = normalizar(retas[8])
	linM5 = normalizar(retas[9])
	print(linL1, linM1, linL2, linM2, linL3, linM3)
	m = array([[linL1[0]*linM1[0], (linL1[0]*linM1[1] + linL1[1]*linM1[0])/2, linL1[1]*linM1[1], (linL1[0]*linM1[2] + linL1[2]*linM1[0])/2, (linL1[1]*linM1[2] + linL1[2]*linM1[1])/2, linL1[2]*linM1[2]],
				[linL2[0]*linM2[0], (linL2[0]*linM2[1] + linL2[1]*linM2[0])/2, linL2[1]*linM2[1], (linL2[0]*linM2[2] + linL2[2]*linM2[0])/2, (linL2[1]*linM2[2] + linL2[2]*linM2[1])/2, linL2[2]*linM2[2]],
				[linL3[0]*linM3[0], (linL3[0]*linM3[1] + linL3[1]*linM3[0])/2, linL3[1]*linM3[1], (linL3[0]*linM3[2] + linL3[2]*linM3[0])/2, (linL3[1]*linM3[2] + linL3[2]*linM3[1])/2, linL3[2]*linM3[2]],
				[linL4[0]*linM4[0], (linL4[0]*linM4[1] + linL4[1]*linM4[0])/2, linL4[1]*linM4[1], (linL4[0]*linM4[2] + linL4[2]*linM4[0])/2, (linL4[1]*linM4[2] + linL4[2]*linM4[1])/2, linL4[2]*linM4[2]],
				[linL5[0]*linM5[0], (linL5[0]*linM5[1] + linL5[1]*linM5[0])/2, linL5[1]*linM5[1], (linL5[0]*linM5[2] + linL5[2]*linM5[0])/2, (linL5[1]*linM5[2] + linL5[2]*linM5[1])/2, linL5[2]*linM5[2]],
				[0, 0, 0, 0, 0, 1]])
	detM = det(m)
	print("Determinante de m", detM)
	s = zeros(6)
	res = array([0, 0, 0, 0, 0, 1])
	for i in range(6):
		mI = copy(m)
		mI[:,i] = res
		print("Determinante", i, det(mI))
		s[i] = det(mI) / detM
	conica = array([[s[0], s[1], s[3]],
						[s[1], s[2], s[4]],
						[s[3], s[4], s[5]]])
	print("Conica:")
	print(conica)
	(u, d, _) = svd(conica)
	print("Transformacao:")
	mD = array([[sqrt(d[0]), 0, 0], [0, sqrt(d[1]), 0], [0, 0, 1]])
	saida = dot(u, mD)
	print(saida)
	print("Conica reconstruida:")
	temp = zeros((3,3))
	temp[0,0] = 1
	temp[1,1] = 1
	print(dot(dot(saida, temp), transpose(saida)))
	return inv(saida)
	
def normalizar(ponto):
	'''
	Deixa o vetor homogeneo com z = 1
	'''
	ponto[0] = ponto[0] / ponto[2]
	ponto[1] = ponto[1] / ponto[2]
	ponto[2] = 1
	return ponto

def get_bounding_box(t, columns, rows):
	'''
	Obtem os limites minimo e maximo de cada coordenada da imagem.
	'''
	ponto1 = normalizar(dot(t, transpose(array([0, 0, 1]))))
	ponto2 = normalizar(dot(t, transpose(array([columns-1, 0, 1]))))
	ponto3 = normalizar(dot(t, transpose(array([columns-1, rows-1, 1]))))
	ponto4 = normalizar(dot(t, transpose(array([0, rows-1, 1]))))
	minX = min(ponto1[0], ponto2[0], ponto3[0], ponto4[0])
	minY = min(ponto1[1], ponto2[1], ponto3[1], ponto4[1])
	maxX = max(ponto1[0], ponto2[0], ponto3[0], ponto4[0])
	maxY = max(ponto1[1], ponto2[1], ponto3[1], ponto4[1])
	return (maxX, minX, maxY, minY)
	
def novas_dimensoes(boundBox, larguraMax, alturaMax):
	'''
	Obtem as dimensoes que a imagem transformada tera.
	'''
	(maxX, minX, maxY, minY) = boundBox
	escala = min(larguraMax / (maxX - minX), alturaMax / (maxY - minY))
	return ((maxX - minX) * escala, (maxY - minY) * escala, escala)

def transf_escala_translacao(boundBox, escala):
	'''
	Gera a matriz que redimensiona e translada a imagem para excaixa-la no boundBox.
	'''
	(_, minX, _, minY) = boundBox
	t = zeros((3, 3))
	t[2,2] = 1
	#Escala
	t[0,0] = escala
	t[1,1] = escala
	#Translacao
	t[0,2] = -minX*escala
	t[1,2] = -minY*escala
	print("Escala e translacao:")
	print(t)
	return t

def produzir_imagem(img, t, largura, altura):
	'''
	Gera a imagem usando a imagem original e a transformacao necessaria.
	'''
	novaImg = zeros((largura, altura, 3), dtype=uint8)
	t_inversa = inv(t)
	for y in range(altura):
		for x in range(largura):
			[xO, yO, _] = normalizar(dot(t_inversa, array([x,y,1])))
			if ((xO >= 0) and (xO < columns) and (yO >= 0) and (yO < rows)):
				novaImg[x, y, :] = img[int(xO), int(yO), :]
	return novaImg

nomeImg = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"), ("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"), ("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])
nomeImg = nomeImg[nomeImg.find("/Imagens/") + 1:]
print("Imagem selecionada:", nomeImg)
img = imread(nomeImg)
img = ajustar_imagem(img)
(columns, rows, _) = img.shape
while (1):
	print("Escolha uma opcao:")
	print("1 - Retificacao com 4 pontos")
	print("2 - Retificacao com a reta do infinito")
	print("3 - Retificacao com dois pares de retas paralelas")
	print("4 - Retificacao com dois pares de retas ortogonais")
	print("5 - Retificacao com cinco pares de retas ortogonais")
	print("0 - Escolher nova imagem")
	opcao = int(input())
	if opcao == 0:
		nomeImg = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"), ("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"), ("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])
		nomeImg = nomeImg[nomeImg.find("/Imagens/") + 1:]
		print("Imagem selecionada:", nomeImg)
		img = imread(nomeImg)
		img = ajustar_imagem(img)
		(columns, rows, _) = img.shape
	else:
		img = transposta(img)
		if opcao == 1:
			dummy = img.copy()
			pontos = []
			def choose_points(event,x,y,flags,param):
				global pontos
				if len(pontos) < 4:
					if event == EVENT_LBUTTONDBLCLK:
						if len(pontos):
							line(img,(pontos[-1][0],pontos[-1][1]),(x,y),(255,0,0),2)
						pontos.append([x,y])
						circle(img,(x,y),5,(0,0,255),-1)
						if len(pontos) == 4:
							line(img,(pontos[0][0],pontos[0][1]),(pontos[-1][0],pontos[-1][1]),(255,0,0),2)
			namedWindow("Antiga")
			setMouseCallback("Antiga",choose_points)
			while (1):
				imshow("Antiga", img)
				k = waitKey(1) & 0xFF
				if k == 27:
					break
			t = ret_afim_4_pontos(pontos[0], pontos[1], pontos[2], pontos[3])
		elif opcao == 2:
			print("Insira os parametros da reta")
			xReta = float(input())
			yReta = float(input())
			zReta = float(input())
			t = ret_afim_reta_inf(array([xReta, yReta, zReta]))
		elif opcao == 3:
			dummy = img.copy()
			pontos = []
			retas = []
			def choose_points(event,x,y,flags,param):
				global pontos, retas
				if len(pontos) < 4:
					if event == EVENT_LBUTTONDBLCLK:
						if len(pontos) % 2:
							line(img,(pontos[-1][0],pontos[-1][1]),(x,y),(255,0,0),2)
							retas.append(cross(pontos[-1], [x,y,1]))
						if len(pontos) >= 2:
							line(img,(pontos[len(pontos)-2][0],pontos[len(pontos)-2][1]),(x,y),(0,255,0),2)
							retas.append(cross(pontos[len(pontos)-2], [x,y,1]))
						pontos.append(array([x,y,1], dtype=float32))
						circle(img,(x,y),5,(0,0,255),-1)
			namedWindow("Antiga")
			setMouseCallback("Antiga",choose_points)
			while (1):
				imshow("Antiga", img)
				k = waitKey(1) & 0xFF
				if k == 27:
					break
			t = ret_afim_reta_inf(cross(cross(retas[0], retas[2]), cross(retas[1], retas[3])))
		elif opcao == 4:
			dummy = img.copy()
			pontos = []
			retas = []
			def choose_points(event,x,y,flags,param):
				global pontos, retas
				if len(retas) < 4:
					if event == EVENT_LBUTTONDBLCLK:
						if len(pontos):
							line(img,(pontos[0][0],pontos[0][1]),(x,y),(255,0,0),2)
						pontos.append(array([x, y, 1], dtype=float32))
						circle(img,(x,y),5,(0,0,255),-1)
						if len(pontos) == 3:
							print(pontos[0], pontos[1], pontos[2])
							retas.append(cross(pontos[0], pontos[1]))
							retas.append(cross(pontos[0], pontos[2]))
							pontos = []
			namedWindow("Antiga")
			setMouseCallback("Antiga",choose_points)
			while (1):
				imshow("Antiga", img)
				k = waitKey(1) & 0xFF
				if k == 27:
					break
			t = ret_retas_ortogonais(retas[0], retas[1], retas[2], retas[3])
		elif opcao == 5:
			dummy = img.copy()
			pontos = []
			retas = []
			def choose_points(event,x,y,flags,param):
				global pontos, retas
				if len(retas) < 10:
					if event == EVENT_LBUTTONDBLCLK:
						if len(pontos):
							line(img,(pontos[0][0],pontos[0][1]),(x,y),(255,0,0),2)
						pontos.append(array([x, y, 1], dtype=float32))
						circle(img,(x,y),5,(0,0,255),-1)
						if len(pontos) == 3:
							print(pontos[0], pontos[1], pontos[2])
							retas.append(cross(pontos[0], pontos[1]))
							retas.append(cross(pontos[0], pontos[2]))
							pontos = []
			namedWindow("Antiga")
			setMouseCallback("Antiga",choose_points)
			while (1):
				imshow("Antiga", img)
				k = waitKey(1) & 0xFF
				if k == 27:
					break
			t = ret_retas_ortogonais_2(retas)
		else:
			print("Opcao invalida")
			img = transposta(dummy.copy())
			continue
		img = transposta(dummy.copy())
		boundBox = get_bounding_box(t, columns, rows)
		print("BOUNDING BOX:")
		print(boundBox)
		larguraMax = 1000
		alturaMax = 600
		largura, altura, escala = novas_dimensoes(boundBox, larguraMax, alturaMax)
		print("LARGURA, ALTURA, ESCALA")
		print(largura, altura, escala)
		t2 = transf_escala_translacao(boundBox, escala)
		print("TESTANDO TRANSFORMACAO")
		print(normalizar(dot(dot(t2, t), array([0, 0, 1]))))
		print(normalizar(dot(dot(t2, t), array([columns-1, rows-1, 1]))))
		print("Transformacao final")
		print(dot(t2, t))
		novaImg = produzir_imagem(img, dot(t2, t), int(largura), int(altura))
		imshow("Antiga", transposta(img))
		imshow("Nova imagem", transposta(novaImg))
		waitKey(0)
		destroyAllWindows()
		while(1):
			print("Deseja salvar a imagem:")
			print("1 - Sim")
			print("2 - Nao")
			opcao = int(input())
			if (opcao == 1):
				imwrite(nomeImg[:-4] + "_retificado.png", novaImg)
				print("Imagem salva como: " + nomeImg[:-4] + "_retificado.png")
				break
			elif (opcao == 2):
				break
			else:
				print("Opcao invalida")