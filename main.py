import perceptron
import random
import neural_network_def
import numpy as np

#Testing perceptron with classifying points to be above or below y=x

class point:
    def  __init__(self):
        self.x = random.uniform(-5,5)
        self.y = random.uniform(-5,5)
        self.label = 1 if self.x>self.y else -1

test_points = []
for i in range (0,1000):
    test_point = point()
    test_points.append(test_point)

main_perceptron = perceptron.perceptron()
main_perceptron.output([1,2])

for point in test_points:
    inputs=[point.x,point.y]
    main_perceptron.train(inputs,point.label)



print(main_perceptron.weights)
total=0
correct=0
for i in range(0,10):
    x= random.uniform(-5,5)
    y=  random.uniform(-5,5)
    result = main_perceptron.output([x,y])
    if (result ==  1 and x>y) or (result==-1 and x<y):
        correct+=1
    total+=1
    print("x is: " + str(x) + " y is: " + str(y))
    print("x>y so 1 expected") if x>y else print("x<y so -1 expected")
    print(main_perceptron.output([x,y]))


print(str(correct/total)) #This outputs the accuracy of the perceptron's classfications

#Testing neural_network with simple xor outputs
nn = neural_network_def.neural_network(2,2,1)
inputs_targets= [[[0,0],[0]],[[0,1],[1]],[[1,0],[1]],[[1,1],[0]]]
for i in range(0,100000):
    n = random.randint(0,3)
    elem=inputs_targets[n]
    nn.train(elem[0],elem[1])

print(nn.feedForward([0,0]))#Expect 0
print(nn.feedForward([0,1]))#Expect 1 
print(nn.feedForward([1,0]))#Expect 1 
print(nn.feedForward([1,1]))#Expect 0  





