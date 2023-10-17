# Enhanced PRETSA

This repository contains the source code for two security features built on top of PRETSA.

## Context

The primary objective of the PRETSA algorithm is to sanitize event logs for the purpose of process mining. The output includes a set of activities, case identifiers, event numbers, and sensitive execution attributes. The primary sensitive attribute in the original repository was the duration of each activity. The implemented features are designed to significantly enhance the privacy of the output and the sensitive attribute.

### L-Diversity for Event Logs

An event log satisfies l-diversity if, for each equivalent class, such as the executed activities, there are at least `l` distinct values for the sensitive column. By introducing l-diversity to an event log, it becomes challenging for adversaries to extract information from the similarity of the sensitive column. Furthermore, it effectively prevents homogeneity attacks. For instance, if an adversary observes that the sensitive column is the same for one specific activity in several traces, they can deduce that the same employee was involved in multiple sequences of activities within the event log. The adversary can exploit this information, combined with their background knowledge, to identify the individual.

L-diversity is enforced in the PRETSA algorithm during the tree pruning stage. The `_treePrunning` function has been modified to check whether l-diversity is satisfied by invoking the `__violatesLDiversity`.  
Implementation code available in `pretsa.py`
# ε-Differential Privacy  
PRETSA mainly focuses on the traces and variants in the event log, and it does not guarantee privacy for the sensitive values. In real-world contexts, companies may not trust third-party data analysis firms with their raw execution data. In such cases, differential privacy can be a valuable solution. By applying differential privacy to the event log, it is possible to avoid disclosing the exact values while maintaining the data utility necessary for accurate process mining.

In this implementation, IBM's diffprivlib is utilized to add bounded Laplace noise to the query. If `getPrivatisedEventLog` is called with the correct parameters, it invokes `__DPmechanismSetup`. This function initializes the numerical mechanism by providing epsilon, sensitivity, upper bound, and lower bound to diffprivlib. Subsequently, `__applyDiffPrivacy` is called to use the numerical mechanism to add noise to the query.  

Implementation code available in `pretsa.py`

# Elevating the Framework
![img_1.png](https://raw.githubusercontent.com/OmidAfroozeh/EnhancedPRETSA_DPTAssignment2/master/img_1.png)  
*Diagram showcasing addition of the new features*  


The implemented features are not intended to be used concurrently, as applying differential privacy to event logs alone should suffice for ensuring privacy. However, in various contexts, the l-diversity feature can also be considered as an option.
# Demos
There are two jupyter notebook available in the repository showcasing the features:  

`Demo_Sepsis_DP.ipnyb` contains code for the Sepsis event log and the differential privacy feature  
`Demo_CoSeLog_diversity.ipnyb` contains code for the CoSeLog event log and showcases the l-diversity feature
# Usage
The features were implemented in Python 3.9.6.  
Beforehand make sure to upgrade pip to the latest version:  
`pip3 install --upgrade pip`  
To install dependencies:  
``pip3 install -r requirements.txt``  
There might be issues installing the pm4py library. If so, try installing it with:  
`pip3 install -U pm4py`


# References
S. A. Fahrenkrog-Petersen, H. van der Aa and M. Weidlich, "PRETSA: Event Log Sanitization for Privacy-aware Process Discovery," 2019 International Conference on Process Mining (ICPM), Aachen, Germany, 2019, pp. 1-8, doi: 10.1109/ICPM.2019.00012.

