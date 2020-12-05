
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2 as cv
import time
import os 
import random
  

num_cell_types=2
pixels_in_each_image=2500
dim_image=int(np.sqrt(pixels_in_each_image))



lr = np.arange(num_cell_types)
hot_list=[]
for label in range(num_cell_types):
    one_hot = (lr==label).astype(np.int)
    hot_list.append(one_hot)
 
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  find_angle(result)
  return result

def find_angle(image):
   contours,hierarchy = cv.findContours(image.astype(np.uint8), 1,method= cv.CHAIN_APPROX_NONE)
   cnt = contours[0]
   rect = cv.minAreaRect(cnt)
   angle=rect[2]
   print(rect[2])

   return(angle)
      
                   


def sigmoid(x):
    s=1/(1+np.exp(-x))
    return(s)

def dsigmoid(x):
    
    s=x*(1-x)
    return(s)

def identify_cells(cell_folder,label):
    
    photos=[]
    basepath="/Users/thibautgold/Documents/Cell_Stamp_album/"+cell_folder#"Shapes"
    for entry in os.listdir(basepath): #Read all photos
        if os.path.isfile(os.path.join(basepath, entry)):
            photos.append(entry)
    
    _list=[]
    _cells=[]
    
    for tiff_index in range(len(photos)): 
        if photos[tiff_index]!='.DS_Store':
            _photo=cv.imread(basepath+"/"+photos[tiff_index],cv.IMREAD_GRAYSCALE)
            _list.append(_photo)
    

    
    for photo in _list[:10]:
        img1 = cv.Canny(photo,100,200)
        img2=cv.blur(img1,(9,9),10000)
        img3=cv.threshold(img2,0,255,cv.THRESH_BINARY)[1]
        
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats((img3), connectivity=8)
       # photo= np.pad(photo, pad_width=50, mode='constant', constant_values=0)    
        
        for i in range(1, nb_components-2):

            if stats[i][-1]>=900:
                
                blank=np.zeros(np.shape(output))
                blank[output==i]=1
                photo2=blank*photo
                photo3= np.pad(photo2, pad_width=50, mode='constant', constant_values=0)    
         
                
                x1=int(centroids[i][0]-max(stats[i,3],stats[i,2])//2)
                x2=int(centroids[i][0]+max(stats[i,3],stats[i,2])//2)
                y1=int(centroids[i][1]-max(stats[i,3],stats[i,2])//2)
                y2=int(centroids[i][1]+max(stats[i,3],stats[i,2])//2)
            
              
                img4=np.zeros(np.shape(output))
                img4[output==i]=1
                angle=find_angle(img4)
                

                
                img5=cv.resize((photo3[y1+50:y2+50, x1+50:x2+50]),(dim_image,dim_image))
                plt.imshow(img5)
                plt.show() 
                
                img6=rotate_image(img5,angle)
                plt.imshow(img6)
                plt.show() 
               # img7 = cv.Canny(img6,50,200)
                
                img8=img6.reshape(pixels_in_each_image,)
                
                
                _cells.append((img8,label))
      
        
    return _cells


 



class NeuralNetwork: 
    def __init__(self,image_pixels,nb_neuron_layer1,nb_neuron_layer2):
        self.nb_neurons_a1=image_pixels
        self.nb_neurons_a2=nb_neuron_layer1
        self.nb_neurons_a3=nb_neuron_layer2
        self.nb_neurons_a4=num_cell_types
        self.success_rate=0
        self.tested=0
        self.success=0
        self.cost_list=[]
        self.cost_average=0
        self.best_performance=0
        self.success_rate_list=[]

        self.initialize()      
       # self.phoenix()


    def initialize(self):
        self.w1=2*np.random.rand(self.nb_neurons_a2,self.nb_neurons_a1)-1

        
        self.w2=2*np.random.rand(self.nb_neurons_a3,self.nb_neurons_a2)-1

        
        self.w3=2*np.random.rand(self.nb_neurons_a4,self.nb_neurons_a3)-1

        
    def phoenix(self):
        self.nb_neurons_a2=np.shape(w1)[0]
        self.nb_neurons_a3=np.shape(w2)[0]

        self.w1=w1.astype(np.float128)
        self.w2=w2.astype(np.float128)
        self.w3=w3.astype(np.float128)
        
        self.b1=np.zeros(self.nb_neurons_a2).astype(np.float128)
        self.b2=np.zeros(self.nb_neurons_a3).astype(np.float128)
        self.b3=np.zeros(self.nb_neurons_a4).astype(np.float128)
    

        
        
    def run(self,input_vector,iterations,mutation_rate,ignore_correct_ans=False):
        
        self.input_vector=[a_tuple[0] for a_tuple in input_vector]
        self.train_label=[a_tuple[1] for a_tuple in input_vector]
        
      
        
        for i in range(iterations):
            
            self.a1=self.input_vector[i]
            self.z2=np.matmul(self.w1,self.a1)
            self.a2=sigmoid(self.z2)
            
            self.z3=np.matmul(self.w2,self.a2)
            self.a3=sigmoid(self.z3)
            
            self.z4=np.matmul(self.w3,self.a3)
            self.a4=sigmoid(self.z4)
            
            
            if ignore_correct_ans==True:
                if  np.argmax(self.a4)!=int(self.train_label[i]):
                    
                    self.error_a4=((hot_list[int(self.train_label[i])]-self.a4)**1)/1         
                    self.delta_a4=self.error_a4*dsigmoid(self.a4)
                    
                    self.error_a3=np.matmul(self.delta_a4,self.w3)
                    self.delta_a3=self.error_a3*dsigmoid(self.a3)
                    
                    self.error_a2=np.matmul(self.delta_a3,self.w2)
                    self.delta_a2=self.error_a2*dsigmoid(self.a2)
                    
               
        
                    self.w3+=self.a3*(self.delta_a4.reshape(num_cell_types,1))*mutation_rate
                    self.w2+=self.a2*(self.delta_a3.reshape(self.nb_neurons_a3,1))*mutation_rate
                    self.w1+=self.a1*(self.delta_a2.reshape(self.nb_neurons_a2,1))*mutation_rate
                
                
            if ignore_correct_ans==False:             
                self.error_a4=((hot_list[int(self.train_label[i])]-self.a4)**1)/1         
                self.delta_a4=self.error_a4*dsigmoid(self.a4)
                
                self.error_a3=np.matmul(self.delta_a4,self.w3)
                self.delta_a3=self.error_a3*dsigmoid(self.a3)
                
                self.error_a2=np.matmul(self.delta_a3,self.w2)
                self.delta_a2=self.error_a2*dsigmoid(self.a2)
                
           
    
                self.w3+=self.a3*(self.delta_a4.reshape(num_cell_types,1))*mutation_rate
                self.w2+=self.a2*(self.delta_a3.reshape(self.nb_neurons_a3,1))*mutation_rate
                self.w1+=self.a1*(self.delta_a2.reshape(self.nb_neurons_a2,1))*mutation_rate
     
    
    def test_performance(self,input_vector1,iterations):
        
        self.input_vector=[a_tuple[0] for a_tuple in input_vector1]
        self.train_label=[a_tuple[1] for a_tuple in input_vector1]   
        self.success=0
        
        
        for i in range(iterations):
            self.a1=self.input_vector[i]
            self.z2=np.matmul(self.w1,self.a1)
            self.a2=sigmoid(self.z2)
            
            self.z3=np.matmul(self.w2,self.a2)
            self.a3=sigmoid(self.z3)
        
            self.z4=np.matmul(self.w3,self.a3)
            self.a4=sigmoid(self.z4)
        
          #  print(self.a4)
         #   print("Guess:", np.argmax(self.a4),"Answer:",int(train_label[i]))
          #  img = input_vector[i].reshape((28,28))
           # plt.imshow(img, cmap="Greys")
          #  plt.show()
          #  print("Guess:",np.argmax(self.a4),"Answer:",train_label[i])
            
            if  np.argmax(self.a4)==int(self.train_label[i]):
                self.success+=1
          #  else:
          #      print(self.a4)
            #    print(np.argmax(self.a4),int(train_label[i]))
        if (self.success/iterations)*100>=self.best_performance:
            self.best_performance=(self.success/iterations)*100
            self.best_wheights=(self.w1,self.w2,self.w3,self.success/iterations*100)

            
        print("Current Success Rate:",(self.success/iterations)*100,"%",self.a4)
        self.success_rate_list.append((self.success/iterations*100))
        
    def test_my_numbers(self):
        input_vector=trymynumbers()
        for i in range(len(input_vector)):
            self.a1=input_vector[i]
            self.z2=np.matmul(self.w1,self.a1)-self.b1
            self.a2=sigmoid(self.z2)
            
            self.z3=np.matmul(self.w2,self.a2)-self.b2
            self.a3=sigmoid(self.z3)
            
            self.z4=np.matmul(self.w3,self.a3)-self.b3
            self.a4=sigmoid(self.z4)
        
          #  print(self.a4)
         
            img = input_vector[i].reshape((28,28))
            plt.imshow(img, cmap="Greys")
            plt.show()
            print("Guess:", np.argmax(self.a4),"confidence:",np.max(self.a4)*100,"%")

BRAIN=NeuralNetwork(pixels_in_each_image,500,50) 

    
liver_cells=identify_cells("LIVER/",0)
pancreas_cells=identify_cells("PANCREAS/",1)


training_set=liver_cells+pancreas_cells
random.shuffle(training_set)

for j in range(1000):
    BRAIN.test_performance(training_set,len(training_set))
    for i in range(1):
        BRAIN.run(training_set,len(training_set),1,False)
    
        

    