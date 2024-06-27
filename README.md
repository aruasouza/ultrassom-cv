# Segmentação de órgãos e estruturas fetais em exames ultrassonográficos



## Descrição:

Para esta aplicação foi utilizado o dataset presente [neste link](https://data.mendeley.com/datasets/4gcpm9dsc3/1), com por volta de 1500 imagens coletadas de 169 participantes que contribuíram com um número váriavel de imagens da circunferência do abdômen fetal (CA). As participantes elegíveis foram gestantes com idade igual ou superior a 18 anos, em trabalho de parto ou com parto programado na Maternidade do Hospital Universitário Polydoro Ernani de São Thiago, em Florianópolis, Santa Catarina, Brasil. 

Médicos especialistas usaram um protocolo de aquisição padronizado para obter imagens de ultrassom. Uma seção axial comum da CA fetal foi capturada, com medidas tomadas na parte mais larga do abdômen fetal, abrangendo o fígado. Esta seção abrangeu o estômago fetal, a artéria aorta, a coluna vertebral e a porção intra-hepática da veia umbilical.

As imagens  de ultrassom foram adquiridas usando uma variedade de dispositivos de ultrassom. Para o processo de anotação, foi empregado o software 3D Slicer para marcar e identificar cuidadosamente cada região de interesse e condição, como esteatose hepática e desequilíbrio metabólico. Depois que as imagens e anotações ficaram prontas, converteu-se as imagens de ultrassom para o formato PNG para facilitar a manipulação e análise. As anotações foram exportadas do 3D Slicer e salvas no formato .npy. Dentro de cada arquivo .npy, é incluído um dicionário que mapeia as estruturas anotadas para suas respectivas imagens. 

### Inicializações:


```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pandas as pd
from modelo import load,Model_cos,Model_2,ORGAOS, colors,clean,create_reshape,transform,reduce, remove_black, find_IOU
from plot import img_with_labels, plot_pred
import seaborn as sns
import cv2
import random
```


```python
def plot_images(data_ex, title, with_labels=False):
    print(title)
    plt.figure(figsize=(30, 30))
    for i in range(len(data_ex)):
        plt.subplot(5, 5, i + 1)
        img, structures = data_ex[i]
        if with_labels:
            img = img_with_labels(img, structures)
        plt.imshow(img, 'gray' if not with_labels else None)
        plt.axis("off")
    
    if with_labels:
        legend_patches = [mpatches.Patch(color=color, label=label) for label, color in zip(ORGAOS, [cor/255 for cor in list(colors.values())])]
        plt.legend(handles=legend_patches, loc='upper left', fontsize=18, bbox_to_anchor=(1.1, 1))
    
    plt.show()
```

### Visão geral do dataset:


```python
folder = 'ARRAY_FORMAT'
data_files = [os.path.join(folder,name) for name in os.listdir(folder)]
print(f"{len(data_files)} Imagens no total.\n")

data_ex = []

for i in range(5):
    img, structures = load(data_files[random.randint(0, len(data_files))]) 
    data_ex.append((img, structures))
    print('Files:',i,end = '\r')


plot_images(data_ex, "Exemplos de imagens cruas:")

plot_images(data_ex, "Exemplos de imagens com as anotações:", with_labels=True)


```

    1588 Imagens no total.
    
    Exemplos de imagens cruas:



    
![png](execucao_detalhada_files/execucao_detalhada_6_1.png)
    


    Exemplos de imagens com as anotações:



    
![png](execucao_detalhada_files/execucao_detalhada_6_3.png)
    


### Geração do modelo de classificação

A criação do modelo começa com a conversão das estruturas das imagens em vetores para facilitar o processo de clusterização. Esses vetores lineares são então agrupados utilizando o algoritmo K-Means, que organiza as imagens em 10 clusters com base nas similaridades de suas estruturas.

Para cada cluster identificado, é construído um modelo agregado das estruturas anatômicas. Esse modelo é gerado somando as estruturas escuras (artéria, estômago e veia) de todas as imagens pertencentes ao mesmo cluster, resultando em uma representação média das estruturas para aquele grupo específico. Além disso, templates detalhados para cada órgão (como artérias, fígado, estômago e veias) são criados dentro de cada cluster. Esses templates são gerados somando as estruturas correspondentes de cada órgão e, em seguida, normalizando pela quantidade de imagens no cluster.

Essa abordagem permite identificar padrões comuns entre as diferentes imagens e criar representações médias que refletem as características típicas das estruturas anatômicas presentes em cada grupo. Assim, o modelo final é composto por esses agregados médios e templates detalhados, proporcionando uma visão clara das variações estruturais e de representação de imagens entre os grupos.

Foram utilizados 2 métodos diferentes para a classificação de uma dada imagem dentro dos 10 clusters:

- **Cosseno de similaridade**: Para este método, as imagens template de cada classe têm fundo cinza e são subtraídas as estruturas escuras de cada elemento pertencente ao cluster e normalizado pelo número total dele. Isso gera imagens que devem naturalmente se parecer com as imagens que queremos processar.
- **Subtração simples normalizada**: Neste método, os templates começam em um fundo preto e as estruturas escuras são adicionadas e normalizadas. Isto é feito porque o template da classe será subtraído da imagem de interesse, reduzindo a soma total dos pixels do objeto de análise. No entanto, as zonas escuras dos órgãos artéria, estômago e veia que serão subtraídas de um valor e, caso a imagem template seja parecida com a imagem de análise, a soma dos pixels da imagem de subtração será semelhante à imagem em questão, pois as intensidades negativas dos pixels são mantidas em 0.


    

#### Templetes para o modelo que usa subtração para classificar:


```python
folder_class = 'MODEL/classes'
data_files_class = [os.path.join(folder_class,name) for name in sorted(os.listdir(folder_class))]


n = len(data_files_class)
num_rows = 2  
num_cols = n//num_rows


fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))

for img_idx, img_path in enumerate(data_files_class):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    class_number = img_path.split('/')[-1].split('.')[0]
    row_idx = img_idx // num_cols
    col_idx = img_idx % num_cols
    ax = axes[row_idx, col_idx]
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Classe {class_number}', fontsize=12)


plt.tight_layout()
plt.show()
```


    
![png](execucao_detalhada_files/execucao_detalhada_10_0.png)
    


#### Templetes para o modelo que cosseno de similaridade


```python
folder_class_c = 'MODEL_c/classes'
data_files_class_c = [os.path.join(folder_class_c,name) for name in sorted(os.listdir(folder_class_c))]


n = len(data_files_class_c)
num_rows = 2  
num_cols = n//num_rows


fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))

for img_idx, img_path in enumerate(data_files_class_c):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    class_number = img_path.split('/')[-1].split('.')[0]
    row_idx = img_idx // num_cols
    col_idx = img_idx % num_cols
    ax = axes[row_idx, col_idx]
    ax.imshow(img,vmax=255, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Classe {class_number}', fontsize=12)


plt.tight_layout()
plt.show()
```


    
![png](execucao_detalhada_files/execucao_detalhada_12_0.png)
    


### Pipeline de processamento:



#### Classificação:

**Passo 1**: Limpeza das imagens usando utilizando operações morfológicas de erosão e dilatação para processar a máscara e escolher a maior região conectada da imagem.

**Passo 2**: Para padronizar as imagens é feito o reshape.

**Passo 3**: É feita também a remoção das bordas da imagens e retirado o fundo preto.

**Passo 4**: É realizado o calculo de similiariade entre as imagens em questão e os templates de classes do modelo criado com o uso de K-means.


```python
#Passo 1: limpeza 
cleaned_imgs = [(clean(img),stru) for img,stru in data_ex]

plot_images(cleaned_imgs, "Imagens limpas:")

#passo 2: Reshape
scalers = [create_reshape(x) for x,_ in cleaned_imgs]
t_imgs = [(transform(cleaned_imgs[i][0],scalers[i]),cleaned_imgs[i][1]) for i in range(len(cleaned_imgs))]
plot_images(t_imgs ,"Imagens após reshape:")

#passo 3: Transformações
r_imgs = [(reduce(t_imgs[i][0]),t_imgs[i][1]) for i in range(len(t_imgs))]
r_b_imgs = [(remove_black(r_imgs[i][0]),r_imgs[i][1]) for i in range(len(r_imgs))]
plot_images(r_b_imgs,"Após transformações:")

#passo 4: Classificação em classes
model_sub  = Model_2('MODEL')
model_cos =  Model_cos('MODEL_c')
classes_sub = [model_sub.get_best_class(data_ex[i][0]) for i in range(len(data_ex))]
classes_cos= [model_cos.get_best_class(data_ex[i][0]) for i in range(len(data_ex))]


print("Classes classificadas por subtração:",classes_sub)
print("Classes classificadas com cosseno de similaridade:",classes_cos)
```

    Imagens limpas:



    
![png](execucao_detalhada_files/execucao_detalhada_16_1.png)
    


    Imagens após reshape:



    
![png](execucao_detalhada_files/execucao_detalhada_16_3.png)
    


    Após transformações:



    
![png](execucao_detalhada_files/execucao_detalhada_16_5.png)
    


    Classes classificadas por subtração: [8, 2, 8, 8, 2]
    Classes classificadas com cosseno de similaridade: [1, 6, 7, 6, 0]


#### Inferência:

Após a identificação da melhor classe (modelo), são utilizados templates específicos associados aos órgãos para inferir a presença dessas estruturas na imagem.


```python
model_sub = Model_2("MODEL")
model_cos =Model_cos("MODEL_c")
for img,struc in data_ex:

    prediction_sub = model_sub.predict(img)
    prediction_cos = model_cos.predict(img)
    print("Modelo que usa Subtração:")
    plot_pred(img,struc,prediction_sub)
    print("Modelo que usa Cosseno de similaridade:")
    plot_pred(img,struc,prediction_cos)
```

    Modelo que usa Subtração:



    
![png](execucao_detalhada_files/execucao_detalhada_18_1.png)
    


    Modelo que usa Cosseno de similaridade:



    
![png](execucao_detalhada_files/execucao_detalhada_18_3.png)
    


    Modelo que usa Subtração:



    
![png](execucao_detalhada_files/execucao_detalhada_18_5.png)
    


    Modelo que usa Cosseno de similaridade:



    
![png](execucao_detalhada_files/execucao_detalhada_18_7.png)
    


    Modelo que usa Subtração:



    
![png](execucao_detalhada_files/execucao_detalhada_18_9.png)
    


    Modelo que usa Cosseno de similaridade:



    
![png](execucao_detalhada_files/execucao_detalhada_18_11.png)
    


    Modelo que usa Subtração:



    
![png](execucao_detalhada_files/execucao_detalhada_18_13.png)
    


    Modelo que usa Cosseno de similaridade:



    
![png](execucao_detalhada_files/execucao_detalhada_18_15.png)
    


    Modelo que usa Subtração:



    
![png](execucao_detalhada_files/execucao_detalhada_18_17.png)
    


    Modelo que usa Cosseno de similaridade:



    
![png](execucao_detalhada_files/execucao_detalhada_18_19.png)
    


### Resultados:


```python
table_precision = pd.read_csv('results/precision_data.csv', index_col=0)
table_recall = pd.read_csv('results/recall_data.csv', index_col=0)
```


```python
ORGAOS = table_precision.index.tolist()
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
axes[0].bar(ORGAOS, table_precision['Model_2_10classes'], color='blue', alpha=0.5, label='Model 2')
axes[0].bar(ORGAOS, table_precision['Model_Cos_10classes'], color='red', alpha=0.5, label='Model Cos')
axes[0].set_title('Precision Comparison - 10 classes')
axes[0].legend()

axes[1].bar(ORGAOS, table_recall['Model_2_10classes'], color='blue', alpha=0.5, label='Model 2')
axes[1].bar(ORGAOS, table_recall['Model_Cos_10classes'], color='red', alpha=0.5, label='Model Cos')
axes[1].set_title('Recall Comparison - 10 classes')
axes[1].legend()

plt.show()

```


    
![png](execucao_detalhada_files/execucao_detalhada_21_0.png)
    



```python

iou_sums_sub = {org: 0 for org in ORGAOS}
iou_sums_cos = {org: 0 for org in ORGAOS}
iou_counts = {org: 0 for org in ORGAOS}


for i, file in enumerate(data_files):  
    img, struc = load(file)
    prediction_sub = model_sub.predict(img)  
    prediction_cos = model_cos.predict(img)  
    
    for org in ORGAOS:
        if org in struc:
            true_mask = struc[org]
            iou_sub = find_IOU(true_mask, prediction_sub[org])
            iou_cos = find_IOU(true_mask, prediction_cos[org])
            iou_sums_sub[org] += iou_sub
            iou_sums_cos[org] += iou_cos
            iou_counts[org] += 1
    print('Files:',i,end = '\r')


iou_averages_sub = {org: iou_sums_sub[org] / iou_counts[org] if iou_counts[org] > 0 else 0 for org in ORGAOS}
iou_averages_cos = {org: iou_sums_cos[org] / iou_counts[org] if iou_counts[org] > 0 else 0 for org in ORGAOS}

for org, iou_avg in iou_averages_sub.items():
    print(f'Média de IoU para {org} utilizando o modelo baseado em subtração: {iou_avg:.4f}')
print("----------------------------")
for org, iou_avg in iou_averages_cos.items():
    print(f'Média de IoU para {org} utilizando o modelo baseado em cosseno de similaridade: {iou_avg:.4f}')
```

    Média de IoU para artery utilizando o modelo baseado em subtração: 0.0291
    Média de IoU para liver utilizando o modelo baseado em subtração: 0.1736
    Média de IoU para stomach utilizando o modelo baseado em subtração: 0.1044
    Média de IoU para vein utilizando o modelo baseado em subtração: 0.1084
    ----------------------------
    Média de IoU para artery utilizando o modelo baseado em cosseno de similaridade: 0.0396
    Média de IoU para liver utilizando o modelo baseado em cosseno de similaridade: 0.1903
    Média de IoU para stomach utilizando o modelo baseado em cosseno de similaridade: 0.1032
    Média de IoU para vein utilizando o modelo baseado em cosseno de similaridade: 0.1202



```python

organs = list(ORGAOS)
iou_values_sub = [iou_averages_sub[org] for org in organs]
iou_values_cos = [iou_averages_cos[org] for org in organs]


x = np.arange(len(organs)) 
width = 0.35  

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width/2, iou_values_sub, width, label='Model_sub',color = ['#958383'] * len(organs))
bars2 = ax.bar(x + width/2, iou_values_cos, width, label='Model_cos',color = ['#000000'] * len(organs))


ax.set_xlabel('Órgãos')
ax.set_ylabel('IoU Média')
ax.set_title('Comparação de IoU Média entre Model_sub e Model_cos')
ax.set_xticks(x)
ax.set_xticklabels(['Artéria',"Figado", "Estômago","Veia"])
ax.legend()


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)

plt.show()
```


    
![png](execucao_detalhada_files/execucao_detalhada_23_0.png)
