import numpy as np
from numpy.linalg import inv, det
import cv2
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
	# print("Determinante de a =", detA)
	# print(a)
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
	# print(np.dot(t, np.array([ponto1[0], ponto1[1], 1])))
	# print(np.dot(t, np.array([ponto2[0], ponto2[1], 1])))
	# print(np.dot(t, np.array([ponto3[0], ponto3[1], 1])))
	# print(np.dot(t, np.array([ponto4[0], ponto4[1], 1])))
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
	print("5 - Retificacao metrica")
	print("0 - Sair")
	opcao = int(input())
	if opcao == 0:
		break
	else:
		if opcao == 1:
			img = transposta(img)
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
			img = transposta(dummy.copy())
			t = ret_afim_4_pontos(pontos[0], pontos[1], pontos[2], pontos[3])
			
			print("Insira os parametros dos pontos")
			# x1 = float(input("x1 = "))
			# y1 = float(input("y1 = "))
			# x2 = float(input("x2 = "))
			# y2 = float(input("y2 = "))
			# x3 = float(input("x3 = "))
			# y3 = float(input("y3 = "))
			# x4 = float(input("x4 = "))
			# y4 = float(input("y4 = "))
			# t = ret_afim_4_pontos([x1, y1], [x2, y2], [x3, y3], [x4, y4])
		elif opcao == 2:
			print("Insira os parametros da reta")
			xReta = float(input())
			yReta = float(input())
			zReta = float(input())
			t = ret_afim_reta_inf(np.array([xReta, yReta, zReta]))
		elif opcao == 5:
			print("Insira")
		else:
			print("Opcao invalida")
			continue
		boundBox = get_bounding_box(t, columns, rows)
		print("BOUNDING BOX:")
		print(boundBox)
		larguraMax = 800
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

ret_afim_4_pontos([100,100], [0,100], [0,0], [100,0])

ret_afim_reta_inf(np.array([2, 4, 5]))

