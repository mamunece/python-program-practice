#z=input("enter a string:") 

def reverse(x):
    t=x
    p=len(t);
    y=p-1;
    d=""
    
    for c in range(0,p):
        d=d+t[y-c]
        
    print(d)
reverse("1234abcd")


        
    
