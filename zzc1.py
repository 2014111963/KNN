from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
digits = load_digits()
X_train,X_test,Y_train,Y_test = train_test_split(digits.data,digits.target,test_size=0.25,random_state = 33)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
lscv = LinearSVC()
lscv.fit(X_train,Y_train)
y_predict = lscv.predict(X_test)
print("The Accuracy of LinearSVC is",lscv.score(X_test,Y_test))