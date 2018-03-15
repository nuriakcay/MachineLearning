
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

iris = load_iris()

X = iris.data

y = iris.target

#SVM Sınıflandırıcısının Modelinin Oluşturulması
clf = svm.SVC(random_state=0)


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
print("\n-->Test Sonucu: {}".format(test_sonuc))

if test_sonuc == 0:
    print("\nGirmiş olduğunuz bilgiler I.Setosa bitkisine ait \n")
elif test_sonuc == 1:
    print("\nGirmiş olduğunuz bilgiler I.Versicolor bitkisine ait \n")
elif test_sonuc == 2:
    print("\nGirmiş olduğunuz bilgiler I.Virginia bitkisine ait \n")    


    
#Verinin    %70'ini Eğitim, %30'sini test verisi olarak ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.7, test_size = 0.3, random_state=0, stratify = y)

#Eğitim verisi ile eğitimi gerçekleştiriyoruz
clf.fit(X_train,y_train)

test_sonuc = clf.predict(X_test)
print ("Test Sonucu: ",test_sonuc, "\n")

cross_val = cross_val_score(clf,X,y)
print("SVM Cross Validation Score: {} \n".format(cross_val))

cm = confusion_matrix( y_test, test_sonuc)
print ("Confusion Matrix: \n {} \n".format(cm))

#score = clf.score(X_test,y_test)
score = accuracy_score(y_test,test_sonuc)
print("SVM Classifier Doğruluk Değeri: {}".format(score))



