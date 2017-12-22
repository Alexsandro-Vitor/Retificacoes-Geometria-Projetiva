import numpy as np
from numpy.linalg import inv, det, svd
import cv2
from math import sqrt
from tkinter.filedialog import askopenfilename

def ajustar_imagem(img):
	(lin, col, _) = img.shape
	moldura = np.zeros((lin+2, col+2, 3), dtype=np.uint8)
	for r in range(lin):
		for c in range(col):
			moldura[r+1, c+1, 0:3] = img[r, c, 0:3]
	b,g,r = cv2.split(moldura)
	b = np.transpose(b)
	g = np.transpose(g)
	r = np.transpose(r)
	return cv2.merge((b,g,r))

def transposta(img):
	b,g,r = cv2.split(img)
	b = np.transpose(b)
	g = np.transpose(g)
	r = np.transpose(r)
	return cv2.merge((b,g,r))
	
def ret_afim_4_pontos(ponto1, ponto2, ponto3, ponto4):
	#novos pontos
	#x1' = (0, 0)	x2' = (1, 0)	x3' = (1, 1)	x4' = (0, 1)
	a = np.array([[0, 0, 0, -ponto1[0], -ponto1[1], -1, 0, 0, 0],
				[ponto1[0], ponto1[1], 1, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, -ponto2[0], -ponto2[1], -1, 0, 0, 0],
				[ponto2[0], ponto2[1], 1, 0, 0, 0, -ponto2[0], -ponto2[1], -1],
				[0, 0, 0, -ponto3[0], -ponto3[1], -1, ponto3[0], ponto3[1], 1],
				[ponto3[0], ponto3[1], 1, 0, 0, 0, -ponto3[0], -ponto3[1], -1],
				[0, 0, 0, -ponto4[0], -ponto4[1], -1, ponto4[0], ponto4[1], 1],
				[ponto4[0], ponto4[1], 1, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 0, 0, 0, 1]])
	detA = det(a)
	t = np.zeros((3, 3))
	res = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
	for i in range(9):
		aI = np.copy(a)
		aI[:,i] = res
		# Calculando os valores de t usando DLT com regra de Cramer
		t[i//3,i%3] = det(aI) / detA
		# print("Determinante", i, "=", det(aI))
	print("Retificacao:")
	print(t)
	return t

def ret_afim_reta_inf(reta):
	t = np.zeros((3, 3))
	t[0,0] = 1
	t[1,1] = 1
	t[2] = np.copy(reta)
	print("Retificacao:")
	print(t)
	#print(inv(np.transpose(t)))
	#print(np.dot(np.transpose(t), inv(np.transpose(t))))
	#print(np.dot(inv(np.transpose(t)), reta))
	return t

def ret_retas_ortogonais(linL1, linM1, linL2, linM2):
	print(linL1, linM1, linL2, linM2)
	linL1 = normalizar(linL1)
	linM1 = normalizar(linM1)
	linL2 = normalizar(linL2)
	linM2 = normalizar(linM2)
	print(linL1, linM1, linL2, linM2)
	m = np.array(
		[[linL1[0]*linM1[0], linL1[0]*linM1[1] + linL1[1]*linM1[0], linL1[1]*linM1[1]],
		[linL2[0]*linM2[0], linL2[0]*linM2[1] + linL2[1]*linM2[0], linL2[1]*linM2[1]],
		[0, 0, 1]])
	detM = det(m)
	s = np.zeros(3)
	res = np.array([0, 0, 1])
	for i in range(3):
		mI = np.copy(m)
		mI[:,i] = res
		s[i] = det(mI) / detM
	conica = np.array([[s[0], s[1], 0],
						[s[1], s[2], 0],
						[0, 0, 0]])
	print("Conica:")
	print(conica)
	(u, d, _) = svd(conica)
	print("Transformacao:")
	mD = np.array([[sqrt(d[0]), 0, 0], [0, sqrt(d[1]), 0], [0, 0, 1]])
	saida = np.dot(u, mD)
	print(saida)
	print("Conica reconstruida:")
	temp = np.zeros((3,3))
	temp[0,0] = 1
	temp[1,1] = 1
	print(np.dot(np.dot(saida, temp), np.transpose(saida)))
	return inv(saida)

def ret_retas_ortogonais_2(retas):
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
	m = np.array([[linL1[0]*linM1[0], (linL1[0]*linM1[1] + linL1[1]*linM1[0])/2, linL1[1]*linM1[1], (linL1[0]*linM1[2] + linL1[2]*linM1[0])/2, (linL1[1]*linM1[2] + linL1[2]*linM1[1])/2, linL1[2]*linM1[2]],
				[linL2[0]*linM2[0], (linL2[0]*linM2[1] + linL2[1]*linM2[0])/2, linL2[1]*linM2[1], (linL2[0]*linM2[2] + linL2[2]*linM2[0])/2, (linL2[1]*linM2[2] + linL2[2]*linM2[1])/2, linL2[2]*linM2[2]],
				[linL3[0]*linM3[0], (linL3[0]*linM3[1] + linL3[1]*linM3[0])/2, linL3[1]*linM3[1], (linL3[0]*linM3[2] + linL3[2]*linM3[0])/2, (linL3[1]*linM3[2] + linL3[2]*linM3[1])/2, linL3[2]*linM3[2]],
				[linL4[0]*linM4[0], (linL4[0]*linM4[1] + linL4[1]*linM4[0])/2, linL4[1]*linM4[1], (linL4[0]*linM4[2] + linL4[2]*linM4[0])/2, (linL4[1]*linM4[2] + linL4[2]*linM4[1])/2, linL4[2]*linM4[2]],
				[linL5[0]*linM5[0], (linL5[0]*linM5[1] + linL5[1]*linM5[0])/2, linL5[1]*linM5[1], (linL5[0]*linM5[2] + linL5[2]*linM5[0])/2, (linL5[1]*linM5[2] + linL5[2]*linM5[1])/2, linL5[2]*linM5[2]],
				[0, 0, 0, 0, 0, 1]])
	detM = det(m)
	print("Determinante de m", detM)
	s = np.zeros(6)
	res = np.array([0, 0, 0, 0, 0, 1])
	for i in range(6):
		mI = np.copy(m)
		mI[:,i] = res
		print("Determinante", i, det(mI))
		s[i] = det(mI) / detM
	conica = np.array([[s[0], s[1], s[3]],
						[s[1], s[2], s[4]],
						[s[3], s[4], s[5]]])
	print("Conica:")
	print(conica)
	(u, d, _) = svd(conica)
	print("Transformacao:")
	mD = np.array([[sqrt(d[0]), 0, 0], [0, sqrt(d[1]), 0], [0, 0, 1]])
	saida = np.dot(u, mD)
	print(saida)
	print("Conica reconstruida:")
	temp = np.zeros((3,3))
	temp[0,0] = 1
	temp[1,1] = 1
	print(np.dot(np.dot(saida, temp), np.transpose(saida)))
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
	ponto1 = normalizar(np.dot(t, np.transpose(np.array([0, 0, 1]))))
	ponto2 = normalizar(np.dot(t, np.transpose(np.array([columns-1, 0, 1]))))
	ponto3 = normalizar(np.dot(t, np.transpose(np.array([columns-1, rows-1, 1]))))
	ponto4 = normalizar(np.dot(t, np.transpose(np.array([0, rows-1, 1]))))
	minX = min(ponto1[0], ponto2[0], ponto3[0], ponto4[0])
	minY = min(ponto1[1], ponto2[1], ponto3[1], ponto4[1])
	maxX = max(ponto1[0], ponto2[0], ponto3[0], ponto4[0])
	maxY = max(ponto1[1], ponto2[1], ponto3[1], ponto4[1])
	return (maxX, minX, maxY, minY)
	
def novas_dimensoes(boundBox, larguraMax, alturaMax):
	(maxX, minX, maxY, minY) = boundBox
	escala = min(larguraMax / (maxX - minX), alturaMax / (maxY - minY))
	return ((maxX - minX) * escala, (maxY - minY) * escala, escala)

def transf_escala_translacao(boundBox, escala):
	(maxX, minX, maxY, minY) = boundBox
	t = np.zeros((3, 3))
	t[2,2] = 1
	#Escala
	t[0,0] = escala
	t[1,1] = escala
	#Translacao
	t[0,2] = -minX*escala
	t[1,2] = -minY*escala
	#print(np.dot(t, np.array([minY, minX, 1])))
	#print(np.dot(t, np.array([maxY, maxX, 1])))
	print("Escala e translacao:")
	print(t)
	return t

def produzir_imagem(img, t, largura, altura):
	novaImg = np.zeros((largura, altura, 3), dtype=np.uint8)
	t_inversa = inv(t)
	for y in range(altura):
		for x in range(largura):
			[xO, yO, _] = normalizar(np.dot(t_inversa, np.array([x,y,1])))
			if ((xO >= 0) and (xO < columns) and (yO >= 0) and (yO < rows)):
				novaImg[x, y, 0] = img[int(xO), int(yO), 0]
				novaImg[x, y, 1] = img[int(xO), int(yO), 1]
				novaImg[x, y, 2] = img[int(xO), int(yO), 2]
			# if (x == 0 and y == 0):
				# print(img[int(xO), int(yO)])
				# print(novaImg[x, y])
	return novaImg

nomeImg = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"), ("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"), ("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])
nomeImg = nomeImg[nomeImg.find("/Imagens/") + 1:]
print(nomeImg)
img = cv2.imread(nomeImg)
img = ajustar_imagem(img)
(columns, rows, _) = img.shape
while (1):
	print("Escolha uma opcao:")
	print("1 - Retificacao afim com 4 pontos")
	print("2 - Retificacao afim com a reta do infinito")
	print("3 - Retificacao metrica com dois pares de retas ortogonais")
	print("4 - Retificacao com cinco pares de retas ortogonais")
	print("0 - Sair")
	opcao = int(input())
	if opcao == 0:
		break
	else:
		img = transposta(img)
		if opcao == 1:
			dummy = img.copy()
			pontos = []
			def choose_points(event,x,y,flags,param):
				global pontos, dummy
				if len(pontos) < 4:
					if event == cv2.EVENT_LBUTTONDBLCLK:
						if len(pontos):
							cv2.line(img,(pontos[-1][0],pontos[-1][1]),(x,y),(255,0,0),2)
						pontos.append([x,y])
						cv2.circle(img,(x,y),5,(0,0,255),-1)
						if len(pontos) == 4:
							cv2.line(img,(pontos[0][0],pontos[0][1]),(pontos[-1][0],pontos[-1][1]),(255,0,0),2)
			cv2.namedWindow("Antiga")
			cv2.setMouseCallback("Antiga",choose_points)
			print("Evento clique = ", cv2.EVENT_LBUTTONDBLCLK)
			while (1):
				cv2.imshow("Antiga", img)
				k = cv2.waitKey(1) & 0xFF
				if k == 27:
					break
			t = ret_afim_4_pontos(pontos[0], pontos[1], pontos[2], pontos[3])
		elif opcao == 2:
			print("Insira os parametros da reta")
			xReta = float(input())
			yReta = float(input())
			zReta = float(input())
			t = ret_afim_reta_inf(np.array([xReta, yReta, zReta]))
		elif opcao == 3:
			dummy = img.copy()
			pontos = []
			retas = []
			def choose_points(event,x,y,flags,param):
				global pontos, retas, dummy
				if len(retas) < 4:
					if event == cv2.EVENT_LBUTTONDBLCLK:
						if len(pontos):
							cv2.line(img,(pontos[0][0],pontos[0][1]),(x,y),(255,0,0),2)
						pontos.append(np.array([x, y, 1], dtype=np.float32))
						cv2.circle(img,(x,y),5,(0,0,255),-1)
						if len(pontos) == 3:
							print(pontos[0], pontos[1], pontos[2])
							retas.append(np.cross(pontos[0], pontos[1]))
							retas.append(np.cross(pontos[0], pontos[2]))
							pontos = []
			cv2.namedWindow("Antiga")
			cv2.setMouseCallback("Antiga",choose_points)
			print("Evento clique = ", cv2.EVENT_LBUTTONDBLCLK)
			while (1):
				cv2.imshow("Antiga", img)
				k = cv2.waitKey(1) & 0xFF
				if k == 27:
					break
			t = ret_retas_ortogonais(retas[0], retas[1], retas[2], retas[3])
		elif opcao == 4:
			dummy = img.copy()
			pontos = []
			retas = []
			def choose_points(event,x,y,flags,param):
				global pontos, retas, dummy
				if len(retas) < 10:
					if event == cv2.EVENT_LBUTTONDBLCLK:
						if len(pontos):
							cv2.line(img,(pontos[0][0],pontos[0][1]),(x,y),(255,0,0),2)
						pontos.append(np.array([x, y, 1], dtype=np.float32))
						cv2.circle(img,(x,y),5,(0,0,255),-1)
						if len(pontos) == 3:
							print(pontos[0], pontos[1], pontos[2])
							retas.append(np.cross(pontos[0], pontos[1]))
							retas.append(np.cross(pontos[0], pontos[2]))
							pontos = []
			cv2.namedWindow("Antiga")
			cv2.setMouseCallback("Antiga",choose_points)
			print("Evento clique = ", cv2.EVENT_LBUTTONDBLCLK)
			while (1):
				cv2.imshow("Antiga", img)
				k = cv2.waitKey(1) & 0xFF
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
		print(normalizar(np.dot(np.dot(t2, t), np.array([0, 0, 1]))))
		print(normalizar(np.dot(np.dot(t2, t), np.array([columns-1, rows-1, 1]))))
		print("Transformacao final")
		print(np.dot(t2, t))
		novaImg = produzir_imagem(img, np.dot(t2, t), int(largura), int(altura))
		cv2.imshow("Antiga", transposta(img))
		cv2.imshow("Nova imagem", transposta(novaImg))
		cv2.waitKey(0)
		cv2.destroyAllWindows()