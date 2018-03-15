# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

#zambak veri setinin yüklenmesi
iris = load_iris()
#X matrisine 4 sütun olarak 150 bitkinin özelliklerinin aktarılması
X = iris.data
#y dizisine bu 150 bitkinin türlerinin(etiketlerinin) atanması
y = iris.target

#Karar Ağacı Sınıflandırıcısının Modelinin Oluşturulması
# random_state = 0  , sonuçların rastgele değişmemesi için
clf = DecisionTreeClassifier(random_state = 0)

#Verinin    %80'ini Eğitim, %20'sini test verisi olarak ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.8, test_size = 0.2, random_state=0, stratify = y)

#Eğitim verisi ile eğitimi gerçekleştiriyoruz
clf.fit(X_train,y_train) 


print ("\n**************Zambak Çiçeği Tür Sorgulaması İçin 4 Farklı Öznitelik Giriniz")
canakYaprakUz = input("Çanak Yaprak Uzunluğu: ")
canakYaprakGen = input("Çanak Yaprak Genişliği: ")
tacYaprakUz = input("Taç Yaprak Uzunluğu: ")
tacYaprakGen = input("Taç Yaprak Genişliği: ")

kullaniciOznitelikDizisi = (canakYaprakUz,canakYaprakGen,tacYaprakUz,tacYaprakGen)

# Bir adet bitkilik test verisinin hazırlanması
test = ([kullaniciOznitelikDizisi])

#Test verisinin sonucunun, bitkinin türlerinin model üzerinde sınıflandırılması
test_sonuc = clf.predict(test)

#sonucun yazdırılması
print("\n-->Girilmiş Olan Değerler İçin Test Sonucu: {}".format(test_sonuc))

if test_sonuc == 0:
    print("\n-->Girmiş olduğunuz bilgiler I.Setosa bitkisine ait \n")
elif test_sonuc == 1:
    print("\n-->Girmiş olduğunuz bilgiler I.Versicolor bitkisine ait \n")
elif test_sonuc == 2:
    print("\n-->Girmiş olduğunuz bilgiler I.Virginia bitkisine ait \n ")    
    

test_sonuc = clf.predict(X_test)
print ("Test Sonucu: ",test_sonuc, "\n")

cross_val = cross_val_score(clf,X,y)
print("Decision Tree Cross Validation Score: {} \n".format(cross_val))

cm = confusion_matrix( y_test, test_sonuc)
print ("Confusion Matrix: \n {} \n".format(cm))

#score = clf.score(X_test,y_test)
score = accuracy_score(y_test,test_sonuc)
print("Decision Tree Classifier Doğruluk Değeri: {}".format(score))
