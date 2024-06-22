from subprocess import call
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,roc_curve
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

root = tk.Tk()
root.title("HOME PAGE")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
# ++++++++++++++++++++++++++++++++++++++++++++

image2 = Image.open('slide.jpg')

image2 = image2.resize((w, h), Image.LANCZOS)

background_image = ImageTk.PhotoImage(image2)


background_label = tk.Label(root, image=background_image)
background_label.image = background_image



background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)
lbl = tk.Label(root, text="Mobile Botnet Detection", font=('times', 35,' bold '), height=1, width=62,bg="black",fg="white")
lbl.place(x=0, y=0)
# _+++++++++++++++++++++++++++++++++++++++++++++++++++++++



def Model_Training():
    start = time.time()
    data = pd.read_csv("new1.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    data['TelephonyManager.*getDeviceId'] = le.fit_transform(data['TelephonyManager.*getDeviceId'])

    data['TelephonyManager.*getSubscriberId'] = le.fit_transform(data['TelephonyManager.*getSubscriberId'])
    data['abortBroadcast'] = le.fit_transform(data['abortBroadcast'])
    data['SEND_SMS'] = le.fit_transform(data['SEND_SMS'])
    data['DELETE_PACKAGES'] = le.fit_transform(data['DELETE_PACKAGES'])
    data['PHONE_STATE'] = le.fit_transform(data['PHONE_STATE'])
    data['RECEIVE_SMS'] = le.fit_transform(data['RECEIVE_SMS'])
    data['Ljava.net.InetSocketAddress'] = le.fit_transform(data['Ljava.net.InetSocketAddress'])
    data['READ_SMS'] = le.fit_transform(data['READ_SMS'])
    data['android.intent.action.BOOT_COMPLETED'] = le.fit_transform(data['android.intent.action.BOOT_COMPLETED'])
    data['io.File.*delete('] = le.fit_transform(data['io.File.*delete('])
    data['chown'] = le.fit_transform(data['chown'])
    data['chmod'] = le.fit_transform(data['chmod'])
    data['mount'] = le.fit_transform(data['mount'])
    data['.apk'] = le.fit_transform(data['.apk'])
    data['.zip'] = le.fit_transform(data['.zip'])
    data['.dex'] = le.fit_transform(data['.dex'])
    data['CAMERA'] = le.fit_transform(data['CAMERA'])
    data['ACCESS_FINE_LOCATION'] = le.fit_transform(data['ACCESS_FINE_LOCATION'])
    data['INSTALL_PACKAGES'] = le.fit_transform(data['INSTALL_PACKAGES'])
    data['android.intent.action.BATTERY_LOW'] = le.fit_transform(data['android.intent.action.BATTERY_LOW'])
    data['.so'] = le.fit_transform(data['.so'])
    data['android.intent.action.ACTION_POWER_CONNECTED'] = le.fit_transform(data['android.intent.action.ACTION_POWER_CONNECTED'])
    data['System.*loadLibrary'] = le.fit_transform(data['System.*loadLibrary'])
    data['.exe'] = le.fit_transform(data['.exe'])    

    """Feature Selection => Manual"""
    x = data.drop(['ACCESS_NETWORK_STATE','BLUETOOTH','ACCESS_WIFI_STATE','BROADCAST_SMS','CALL_PHONE','CALL_PRIVILEGED','CLEAR_APP_CACHE','CLEAR_APP_USER_DATA','CONTROL_LOCATION_UPDATES','INTERNET','Result'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Result']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,random_state=0)
    #X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = Sequential()
    classifier.add(Dense(activation = "relu", 
                         units = 8, kernel_initializer = "uniform"))
    classifier.add(Dense(activation = "relu", units = 14, 
                         kernel_initializer = "uniform"))
    classifier.add(Dense(activation = "sigmoid", units = 1, 
                         kernel_initializer = "uniform"))
    classifier.add(Dropout(0.2))
    classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', 
                       metrics = ['accuracy'] )
    
    
    classifier.fit(X_train , Y_train , epochs = 100  )
    
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    print("Classification Report :\n")
    repo = (classification_report(Y_test, y_pred))
    print(repo)
    print("\n")
    print("Confusion Matrix :")
    cm = confusion_matrix(Y_test,y_pred)
    print(cm)
    print("\n")
    from mlxtend.plotting import plot_confusion_matrix
 
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Reds)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    accuracy = (cm[0][0]+cm[1][1])/(cm[0][1] + cm[1][0] +cm[0][0] +cm[1][1])
    end = time.time()
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    print(ET)
    print("CNN Accuracy :")
    print(accuracy*100)
    print("\n")
    yes = tk.Label(root,text=accuracy*100,background="green",foreground="white",font=('times', 20, ' bold '),width=15)
    yes.place(x=400,y=400)
    print("Classification Report :\n")
    repo = (classification_report(Y_test, y_pred))
    print(repo)
    print("\n")
    rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(Y_test,y_pred)
    sns.set_style('whitegrid')
    plt.figure(figsize=(10,5))
    plt.title('Reciver Operating Characterstic Curve')

    plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='CSNN Classifier',color='red')  
    plt.plot([0,1],ls='--',color='blue')
    plt.plot([0,0],[1,0],color='green')
    plt.plot([1,1],color='green')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.legend()
    plt.show()



def call_file():
    import Check_botnet
    Check_botnet.Train()


def window():
    root.destroy()

# button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
#                     text="Data_Preprocessing", command=Data_Preprocessing, width=15, height=2)
# button2.place(x=5, y=90)

button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Model Training", command=Model_Training, width=15, height=2)
button3.place(x=5, y=170)

button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Botnet Detection", command=call_file, width=15, height=2)
button4.place(x=5, y=250)

exit = tk.Button(root, text="Exit", command=window, width=15, height=2, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=5, y=330)

root.mainloop()

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''