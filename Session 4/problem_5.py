
def prime(y):
    
    

    f=0
    for x in range(2,y):
        
        
        
        if y%x==0:
            
            f=f+1
        
        
    
    if f>=1:
        
        
        
        return False
    else:
        return True
    



    
        
     

   
       
       
       
c=int(input("Enter a  prime integer:"))
print(prime(c))
