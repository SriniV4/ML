### MODIFY BASED ON OUTPUT

learning_rate = .01
iterations = 10000

###

guess = lambda hypothesis , inp: sum([inp[j] * hypothesis[j] for j in range(len(hypothesis))])
def updateHypothesis(training, hypothesis): # Assume J(h) = sum((h(x_i) - y_i)^2)/2m
        global guess
	# Assume valid input
        features = len(hypothesis)
        m = len(training) # optional extra constant to normalize input -> calculate how much square difference on average 
        for (inp , out) in training:
	        for i in range(features):
		        hypothesis[i] -= (guess(hypothesis , inp) - out) * inp[i] * learning_rate
def getLoss(training , hypothesis): # Assume J(h) = sum((h(x_i) - y_i)^2)/2m
    global guess
    loss = 0
    for (inp , out) in training:
        loss += (guess(hypothesis , inp) - out)**2
    return loss

features = 2 # number of input features + 1 ( first one is always 1 )
training = []
file_name = open("trainingdata.txt" , "r") # Assume valid file: comma seperated features, label
for line in file_name:
    spl = [float(i) for i in line.split(",")]
    features = len(spl)
    training.append([[1] + spl[0:-1] , spl[-1]])

m = len(training) # size of training set
hypothesis = [0 for i in range(features)]

for i in range(iterations):
    loss = getLoss(training,  hypothesis)
    updateHypothesis(training, hypothesis)
    ### COMMENT OUT IF LARGE ITERATIONS
    print("Curr Loss: %.10f" % loss)
    print("New Hypothesis: %s" % hypothesis , end = "\n\n")
    ###

    input() # To Buffer iterations

while(1):
    inp = input("Enter item to predict: ")
    print(guess(hypothesis , [1] + [float(i) for i in inp.split()]))
