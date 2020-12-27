

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    cl=[]
    ### your code goes here

    for i in range (0,len(predictions)):
        tp=(ages[i][0],net_worths[i][0],(predictions[i][0]-net_worths[i][0])**2)
        cl.append(tp[2])
        cleaned_data.append(tp)
    cl.sort(reverse=True)
    cl=cl[:9]
    for i in cleaned_data:
        flag=0
        for j in cl:
            if(i[2]==j):
                flag=1
                break
        if(flag==1):
            cleaned_data.remove(i)
    return cleaned_data

