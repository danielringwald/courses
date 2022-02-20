import numpy as np
from numpy import random as rand
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import math
import scipy.stats as stats
#import statistics


########################
#Problem med att mod einte vet vad den ska göra när det är lika mycket av
#en klass och lägger till 2 värden i klassifieringar

def kontroll_input_typ(meddelande, input_typ = int):
    
    while True:
        try:
            return input_typ(input(meddelande))
        except: 
            print('Ej giltigt svar, skriv igen')
            pass
            
    return 0

def find_mode(arr):
    """Hittar mod i en liste, den 
    
    
    """
    antal_forekomster = np.array([(arr == 1).sum(), (arr == 2).sum(),\
                                  (arr == 3).sum()])
    mode = stats.mode(antal_forekomster)
    
    
    return mode[0] + 1      # +1 då använder index för att bedömma vilken klass 
                            #som förekommer oftast nära 

def kNN_data(antal_data_per_klass, medel_std_matris):
    """Genererar träningsdata
    
    Parametrar: antal_data_per_klass (heltal) 
                    - anger antalet träningsdata som ska genereras per klass.
                medel_std_matris (ndarray) 
                    - en NumPy array med innehåll enligt den principiella struktur som visas i Figur 3 och Figur 4.
    
    Returnerar: (ndarray) 
                    - en NumPy array med data.
    """
    medel_std_matris = np.asarray(medel_std_matris)
    antal_klasser = medel_std_matris.shape[0]
    antal_egenskaper = int((medel_std_matris.shape[1] - 1 ) / 2)
    
    #Allokerar minne till data
    data = np.zeros((antal_data_per_klass*antal_klasser, antal_egenskaper + 1))
    for i in range(antal_klasser):
       data[i*antal_data_per_klass : (i+1)*antal_data_per_klass, 0] = i + 1
       for j in range(antal_egenskaper):
           random_distributed_data = rand.normal(medel_std_matris[i][2*j + 1],\
            medel_std_matris[i][2*j+2], antal_data_per_klass)
           
           data[i*antal_data_per_klass : (i+1)*antal_data_per_klass, j+1] = random_distributed_data
           
    return data

def kNN_algoritm(k, traning_data, okanda_objekt):
    
    okanda_objekt = np.asarray(okanda_objekt)

    if (okanda_objekt.ndim == 1):
        okanda_objekt = np.array([okanda_objekt])
        
    antal_rows_traning_data = traning_data.shape[0]
    antal_okanda_objekt = okanda_objekt.shape[0]
    
    antal_egenskaper = traning_data.shape[1] - 1
   
    klassifieringar = np.empty((0, 1))
    
    for i in range(antal_okanda_objekt):
        
        #FLAGGAR FÖR ATT DET MÅSTE SKICKAS MED okanda_objekt som en ndarray
        okand_data = okanda_objekt[i]       # i:te objektet som ska klassifieras
        #print('Okänd datapunkt')
        #print(okand_data[2])
        avstand = np.empty((0, 2))  #Alla avstånd
        
        for j in range(antal_rows_traning_data):
            
            traning_data_punkt = traning_data[j][1:antal_egenskaper+1]   #j:te traningsdata
            if(antal_egenskaper == 2):
                euklidiskt_avstand = math.sqrt( (okand_data[0] - traning_data_punkt[0])**2 + (okand_data[1] - traning_data_punkt[1])**2 )
            else:
                euklidiskt_avstand = math.sqrt( (okand_data[0] - traning_data_punkt[0])**2 +\
                                    (okand_data[1] - traning_data_punkt[1])**2 + \
                                     (okand_data[2] - traning_data_punkt[2])**2)
                
            avstand = np.append(avstand, np.array([[traning_data[j][0], euklidiskt_avstand]]), axis=0)
            
        
        avstand = pd.DataFrame(avstand, columns = ['klass', 'värde'])
        avstand = avstand.sort_values(by=['värde'])     #Sorterar baserat på euklidiskt avstånd
        k_avstand = avstand.klass[0:k]
       
        mode = find_mode(k_avstand)
        
        klassifieringar = np.append(klassifieringar, mode)
        
       
    return [klassifieringar, avstand.värde[0:k].to_numpy()]
    

def kNN_analysera(k, traning_data, okanda_objekt, sorted_lista, kNN_resultat, korrekt_resultat = None):
    """Den egendefinierade funktionen som analyserar och presenterar en utförd klassificering beskrivs här:
    
    Parametrar: 
        
        k (heltal) - 
        anger antal träningsdata som utgör beslutsgrunden. Om k = 7 bestäms klasstillhörigheten utifrån de 7 träningsdata som har de kortaste avstånden till objektet som ska klassificeras.
    
        traning_data (ndarray) 
            - en Numpy array som innehåller träningsdata enligt den principiella struktur som visas i Figur 1 och Figur 2.
        
        okanda_objekt (ndarray)  
            - en Numpy array som innehåller de okända objekt som ska klassificeras och består av egenskapernas värden. Alltså samma struktur som i Figur 1 och Figur 2 fast där kolumnen med klasstillhörighet (kolumn 1) utelämnats.
        
        sorterad_lista (ndarray)  
            - en Numpy array som innehåller de k stycken kortaste avstånden som beräknades i funktionen kNN_algoritm. 
        
        kNN_resultat (ndarray)  
            - är den NumPy array som returneras av funktionen kNN_algoritm och som innehåller resultatet av en utförd klassificering.
        
        korrekt_resultat (ndarray)
            - en NumPy array som innehåller klasstillhörigheten till de okända objekt som ska klassificeras. 
            - Denna parameter utesluts vid menyalternativ 1. Den får då defaultvärdet None, vilket kan användas i funktionen för att avgöra om utskrift skall ske enligt menyalternativ 1 eller 2.
        
    Returnerar: True om analysen och plottningen lyckades annars False

    """
    plt.close()
    traning_data = np.asarray(traning_data)
    okanda_objekt = np.asarray(okanda_objekt)
    if okanda_objekt.ndim == 1:
        okanda_objekt = np.array([okanda_objekt])
    
    antal_egenskaper = traning_data.shape[1] - 1
    separata_traning_data = np.empty((0, antal_egenskaper))
    
    fig = plt.figure()
    
      
    ### MISSTAGIT LITE, 3D PLOT VID 3 egenskaper men inte 3 klasser, måste kunna hantera 3 klasser också
    if antal_egenskaper == 2:
        ax = fig.add_subplot(111)
        klass_1 = traning_data[(traning_data[:,0] == 1)][:, 1:3]
        klass_2 = traning_data[(traning_data[:,0] == 2)][:, 1:3]
        klass_3 = traning_data[(traning_data[:,0] == 3)][:, 1:3]
        
        x1 = klass_1[:,0]
        y1 = klass_1[:,1]
        x2 = klass_2[:,0]
        y2 = klass_2[:,1]
        x3 = klass_3[:,0]
        y3 = klass_3[:,1]
        
        ax.scatter(x1, y1, c='r')
        ax.scatter(x2, y2, c='b')
        ax.scatter(x3, y3, c='g') 
        plt.title('2D punktdiagram')
        ax.set_xlabel('Egenskap 1')
        ax.set_ylabel('Egenskap 2')
        
    elif antal_egenskaper == 3:
        
        klass_1 = traning_data[(traning_data[:,0] == 1)][:, 1:4]
        klass_2 = traning_data[(traning_data[:,0] == 2)][:, 1:4]
        klass_3 = traning_data[(traning_data[:,0] == 3)][:, 1:4]
        
        x1 = klass_1[:,0]
        y1 = klass_1[:,1]
        z1 = klass_1[:,2]
        x2 = klass_2[:,0]
        y2 = klass_2[:,1]
        z2 = klass_2[:,2]
        x3 = klass_3[:,0]
        y3 = klass_3[:,1]
        z3 = klass_3[:,2]
        
        ax = plt.axes(projection ='3d')
        ax.scatter3D(x1, y1, z1, color = 'r')
        ax.scatter3D(x2, y2, z2, color = 'b')
        ax.scatter3D(x3, y3, z3, color = 'g')
        
        plt.title('3D punktdiagram')
        ax.set_xlabel('Egenskap 1')
        ax.set_ylabel('Egenskap 2')
        ax.set_zlabel('Egenskap 3')
        
    else:
        print('BULL KASTRULL')
    
    if type(korrekt_resultat) == type(None):
        #print(sorted_lista[-1])
        radie = sorted_lista[-1]
        x0 = okanda_objekt[0][0]
        y0 = okanda_objekt[0][1]
        circle = plt.Circle((x0,y0),radie, fill=False)
        ax.add_patch(circle)
        ax.scatter(okanda_objekt[:,0], okanda_objekt[:,1], c='y', marker='s')
        print('Aktuellt k-värde: ' + str(k))
        print('Algoritmen klassificerade det okända okbjektet till klass: ' + str(kNN_resultat[0])) 
    else:   #Mer än en okänd datapunkt
    
        #boolean array där true är korrekt klassifierad     
        jamforelse_klassifiering = (kNN_resultat == korrekt_resultat)[0]   
        fel_lista = np.invert(jamforelse_klassifiering)
        print(fel_lista)
        
        felklassifierade = okanda_objekt[fel_lista]
        korrektklassifierade = okanda_objekt[jamforelse_klassifiering]
        #print(kNN_resultat)
        #print(jamforelse_klassifiering)
        #print(korrektklassifierade)
        #print(felklassifierade)
        ax.scatter(felklassifierade[:,0], felklassifierade[:,1], c='black', marker='s')
        ax.scatter(korrektklassifierade[:,0], korrektklassifierade[:,1], c='y', marker='s')
        print('Aktuellt k-värde: ' + str(k))
        print('Antal utförda klassifieringar: ' + str(kNN_resultat.size))
        print('Antal felklassificeringar: ' + str(felklassifierade[0].size))
        print('Procent felklassificeringar: ' + str(felklassifierade[0].size/kNN_resultat.size * 100))
        
    plt.show()
    
    return True


if __name__ == '__main__':
    run = True
    
    while(run):
        
        print('Klassifiering med hjälp av kNN-algoritmen')
        print('Menyalternativ 1: Klassifiering av 1 okänt objekt med 2 klasser och 2 egenskaper')
        print('Menyalternativ 2: Läser in data från CSV-filer och presenterar dessa i en plot. Visar även precisionen av algoritmen')
        print('Menyalternativ 3: Avslutar programmet')
        menyval = kontroll_input_typ('Menyval: ')
        
        if menyval == 1:
            
            k = kontroll_input_typ('Ange k_värde: ')
            antal_data_per_klass = kontroll_input_typ('Ange antal träningsdata per klass: ')
            mv_k1_e1 = kontroll_input_typ('Medelvärde för klass 1, egenskap 1: ', float)
            std_k1_e1 = kontroll_input_typ('Standardavvikelse för klass 1, egenskap 1: ', float)
            mv_k2_e1 = kontroll_input_typ('Medelvärde för klass 2, egenskap 1: ', float)
            std_k2_e1 = kontroll_input_typ('Standardavvikelse för klass 2, egenskap 1: ', float)
            mv_k1_e2 = kontroll_input_typ('Medelvärde för klass 1, egenskap 2: ', float)
            std_k1_e2 = kontroll_input_typ('Standardavvikelse för klass 1, egenskap 2: ', float)
            mv_k2_e2 = kontroll_input_typ('Medelvärde för klass 2, egenskap 2: ', float)
            std_k2_e2 = kontroll_input_typ('Standardavvikelse för klass 2, egenskap 2: ', float)
            medel_std_matris = [[1, mv_k1_e1, std_k1_e1, mv_k2_e1, std_k2_e1], [2, mv_k2_e1, std_k2_e1, mv_k2_e2, std_k2_e2]]
            x0 = kontroll_input_typ('x-koordinat för okänt datapaket: ', float)
            y0 = kontroll_input_typ('y-koordinat för okänt datapaket: ', float)
            okanda_objekt = [[x0, y0]]
            traning_data = kNN_data(antal_data_per_klass, medel_std_matris)
            [klassifieringar, avstand] = kNN_algoritm(k, traning_data, okanda_objekt)
            kNN_analysera(k, traning_data, okanda_objekt, avstand, klassifieringar)
    
        elif menyval == 2:
            k = kontroll_input_typ('Ange k_värde: ')
            traning_data = input('Ange filnamn för träningsdata: ')
            okanda_objekt = input('Ange filnamn för de okända objekten: ')
            korrekt_resultat = input('Ange filnamn för de korrekt klassificeringarna: ')
            
            traning_data = pd.read_csv(traning_data, header=None)
            okanda_objekt = pd.read_csv(okanda_objekt, header=None)
            korrekt_resultat = pd.read_csv(korrekt_resultat, header=None)
            traning_data = traning_data.to_numpy()
            okanda_objekt = okanda_objekt.to_numpy()
            korrekt_resultat =korrekt_resultat.to_numpy()
            
            [klassificeringar, avstand] = kNN_algoritm(k, traning_data, okanda_objekt)
            kNN_analysera(k, traning_data, okanda_objekt, [0], klassificeringar, korrekt_resultat)
            
        
        elif menyval == 3:
            print('Programmet avslutas.')
            break
        else:
            print("""
                  Du angav ett icke-giltigt val i terminalen, välj mellan
                  alternativ 1, 2 och 3
                  """)
            continue
        
        
        
        
    
    
    

