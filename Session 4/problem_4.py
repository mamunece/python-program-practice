def factorial(x):
    p=1;
    for c in range (0,x):
        p=p*(x-c)
    
    return p
z=int(input("input a number :"))
factorial(z)
c=factorial(z)
print("value of variable :",c)

