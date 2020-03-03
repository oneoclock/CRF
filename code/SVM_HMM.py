import subprocess
import numpy as np
import matplotlib.pyplot as plt

def svm_hmm(c):
    #c = str(c)
    print("Training SVM_HMM...")
    #subprocess.call(['svm_hmm_learn', '-c ', c, 'data/train_struct.txt svm_hmm_model.dat'])
    if c==10:
        subprocess.call('svm_hmm_learn.exe -c 10 ../data/train_struct.txt svm_hmm_model.dat')
    elif c==100:
        subprocess.call('svm_hmm_learn.exe -c 100 ../data/train_struct.txt svm_hmm_model.dat')
    elif c==1000:
        subprocess.call('svm_hmm_learn.exe -c 1000 ../data/train_struct.txt svm_hmm_model.dat')
    elif c==10000:
        subprocess.call('svm_hmm_learn.exe -c 10000 ../data/train_struct.txt svm_hmm_model.dat')
    else:
        print("invalid c value")
    
    subprocess.call('svm_hmm_classify ../data/test_struct.txt svm_hmm_model.dat classify.tags')
    
    FILENAME1 = "classify.tags"
    
    p_tags=np.loadtxt(FILENAME1, delimiter='/n')
    
    FILENAME2 = "../data/test_struct.txt"
    
    tags = np.genfromtxt(FILENAME2, usecols=0)
    
    #Letter wise accuracy
    pred = []
    for i in range(len(p_tags)):
        if p_tags[i] == tags[i]:
            pred.append(True)
        else:
            pred.append(False)
    
    l_acc=sum(pred)/len(pred)
    print("Letter wise Acc:", l_acc)
    
    #obtaining qid from test data set
    test_data = np.genfromtxt(FILENAME2, usecols=1,dtype=str)
         
    # x stores all qids
    x = []
    for i in test_data:
        a=i.split(":")
        a=int(a[1])
        x.append(a)
    
    # letter_count stores length of each word
    num_of_words = x[-1]
    letter_counts = []
    for i in range(1,num_of_words+1):
        letter_counts.append(x.count(i))
    
    #word wise acc calculation
    start=0
    end=0
    pred_result = []
    for letter_count in letter_counts:
        end=end+letter_count
        if np.array_equal(tags[start:end],p_tags[start:end]):
            pred_result.append(True)
        else:
            pred_result.append(False)
        start = end
    
    w_acc=sum(pred_result)/len(pred_result)
    print("Word wise Acc:", w_acc)
    return l_acc, w_acc

def plot_svm_hmm():
    l = []
    w = []
    l_acc, w_acc = svm_hmm(10)
    l.append(l_acc)
    w.append(w_acc)
    #
    l_acc, w_acc = svm_hmm(100)
    l.append(l_acc)
    w.append(w_acc)
    
    l_acc, w_acc = svm_hmm(1000)
    l.append(l_acc)
    w.append(w_acc)

    l_acc, w_acc = svm_hmm(10000)
    l.append(l_acc)
    w.append(w_acc)
    
    c = [10,100,1000,10000]
    plt.figure()
    plt.xlabel('C')
    plt.ylabel('Letter wise accuracy on Test Data')
    plt.plot(c,l)
    plt.show()
    
    plt.figure()
    plt.xlabel('C')
    plt.ylabel('Word wise accuracy on Test Data')
    plt.plot(c,w)
    plt.show()
    return

if __name__=='__main__':
    plot_svm_hmm()
