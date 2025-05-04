# Custom Scania Scoring Function

# - Challenge metric  

#      Cost-metric of miss-classification:

#      Predicted class |      True class       |
#                      |    pos    |    neg    |
#      -----------------------------------------
#       pos            |     -     |  Cost_1   |
#      -----------------------------------------
#       neg            |  Cost_2   |     -     |
#      -----------------------------------------
#      Cost_1 = 10 and cost_2 = 500

# True Positive
# True Negative
# False Positive = cost_1
# False Negative = cost_2

#      The total cost of a prediction model the sum of "Cost_1" 
#      multiplied by the number of Instances with type 1 failure 
#      and "Cost_2" with the number of instances with type 2 failure, 
#      resulting in a "Total_cost".

#      In this case Cost_1 refers to the cost that an unnessecary 
#      check needs to be done by an mechanic at an workshop, while 
#      Cost_2 refer to the cost of missing a faulty truck, 
#      which may cause a breakdown.

#      Total_cost = Cost_1*No_Instances + Cost_2*No_Instances.

def ScaniaScoring(y_true: list, y_pred: list):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = (tn * 10) + (fp * 500)
    ScaniaScoring_score = make_scorer(ScaniaScoring, greater_is_better= False)
    return cost, ScaniaScoring_score
    
    
