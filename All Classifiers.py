
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

class MachineLearning():
    
    def __init__(self,iris):
        self.iris = iris
        self.X = self.iris.data
        self.y = self.iris.target
        #Verinin %70'ini Eğitim, %30'unu test verisi olarak ayırıyoruz
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, train_size = 0.7, test_size =
                                                            0.3, random_state = 0, stratify = self.y)    
        

    def Menu(self):
        print("\n ************************* MENU ***********************")
        print("1-) Test Verisi Girmek İçin 1'e Basınız: ")
        print("2-) Her Classifier İçin Test Sonuçlarını Görmek İçin 2'ye Basınız:")  
        secim = input("Seçenek Giriniz: ")
        
        if(secim == "1"):
            self.UserInput()
        elif(secim == "2"):
            self.RunDecisionTreeClassifier()
            self.RunKNeighborsClassifier()
            self.RunSvm()
            #self.RunMLPClassifier()  
        else:
            print("\nYanlış seşenek girdiniz \n")
            self.Menu()
            
         
    def UserInput(self):
        #Her classifier için farklı bir değişken tanımlanıyor
        clf_decisionTree = DecisionTreeClassifier(random_state=0)
        clf_kNeighbors = KNeighborsClassifier(n_neighbors=2)
        clf_svm = svm.SVC()
           
        #Her classifier'ı elimizde bulunan verilerle eğitiyoruz  
        clf_decisionTree.fit(self.X_train,self.y_train)
        clf_kNeighbors.fit(self.X_train,self.y_train)
        clf_svm.fit(self.X_train,self.y_train)
        
        #Kullanıcıdan eğitilmiş olan classifier türünü test etmesi için 4 farklı öznitelik alıyoruz
        #ve tahminlemeyi bu verilerle yapıyoruz.
        print("************ Kullanıcı Girişi**************")
        print ("\n--->Zambak Çiçeği Tür Sorgulaması İçin 4 Farklı Öznitelik Giriniz")
        canakYaprakUz = input("Çanak Yaprak Uzunluğu: ")
        canakYaprakGen = input("Çanak Yaprak Genişliği: ")
        tacYaprakUz = input("Taç Yaprak Uzunluğu: ")
        tacYaprakGen = input("Taç Yaprak Genişliği: ")
        
        kullaniciOznitelikDizisi = (canakYaprakUz,canakYaprakGen,tacYaprakUz,tacYaprakGen)

        # Bir adet bitkilik test verisinin hazırlanması
        test = ([kullaniciOznitelikDizisi])
         
        #Test verisinin sonucunun, bitkinin türlerinin model üzerinde sınıflandırılması
        test_sonuc_decision = clf_decisionTree.predict(test)
        test_sonuc_kNeighbors = clf_kNeighbors.predict(test)
        test_sonuc_svm = clf_svm.predict(test)
        
        self.UserTestResults('Decision Tree Classifier',test_sonuc_decision)
        self.UserTestResults('KNeighbors Classifier',test_sonuc_kNeighbors)
        self.UserTestResults('SVM Classifier',test_sonuc_svm)
        
        
            
    def UserTestResults(self,algorithm,test_sonuc):
        #sonucun yazdırılması
        print("\n-->",algorithm," İçin Test Sonucu: {}".format(test_sonuc))

        if test_sonuc == 0:
            print("\n-->Girmiş olduğunuz bilgiler I.Setosa bitkisine ait \n")
        elif test_sonuc == 1:
            print("\n-->Girmiş olduğunuz bilgiler I.Versicolor bitkisine ait \n")
        elif test_sonuc == 2:
            print("\n-->Girmiş olduğunuz bilgiler I.Virginia bitkisine ait \n ") 
                         
    def Run(self,algorithm,clf):
        
        # Eğitim Verisi ile eğitimi gerçekleştiriyoruz
        clf.fit(self.X_train,self.y_train)
              
        test_sonuc = clf.predict(self.X_test)
        
        print("\n**************",algorithm," Test Results\n")
        print(test_sonuc)
        
        print("\n",algorithm," Accuracy : " + str(accuracy_score(test_sonuc, self.y_test)))
    
        print("\n",algorithm," Cross Validation Score: \n")
        print(cross_val_score(clf, self.iris.data, self.iris.target, cv=10))
        
        print("\n%s Confusion Matrix: \n" % algorithm)
        cm = confusion_matrix(self.y_test, test_sonuc)
        print(cm)
        
        # Show confusion matrix in a separate window plt.matshow(cm)
        plt.matshow(cm)
        plt.title("%s Confusion Matrix: \n" % algorithm)
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
        
    
    def RunDecisionTreeClassifier(self):  
        clf = DecisionTreeClassifier(random_state=0)
        self.Run('Decision Tree Classifier',clf)
        
    def RunKNeighborsClassifier(self):
        clf = KNeighborsClassifier(n_neighbors=2)
        self.Run('Knn Classifier',clf)    

    def RunSvm(self):
        clf = svm.SVC()
        self.Run('Svm Classifier',clf)
            
    def RunMLPClassifier(self):
        clf =MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=0)
        self.Run('MLP Classifier',clf)
            

        
        
        

        
def main():
    
    iris = load_iris()
    app = MachineLearning(iris)
    app.Menu()
    

    
if __name__ == "__main__":
    main()
