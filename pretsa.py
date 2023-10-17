from anytree import AnyNode, PreOrderIter
from levenshtein import levenshtein
import sys
from scipy.stats import wasserstein_distance
from scipy.stats import normaltest
import pandas as pd
import numpy as np
import time
import diffprivlib as dp

class Pretsa:
    def __init__(self, eventLog):
        # Initializing class variables
        root = AnyNode(id='Root', name="Root", cases=set(), sequence="", annotation=dict(),sequences=set())
        current = root
        currentCase = ""
        caseToSequenceDict = dict()
        sequence = None
        self.__eventlog = eventLog
        self.__caseIDColName = "Case_ID"
        self.__activityColName = "Activity"
        self.__annotationColName = "Duration"
        self.__constantEventNr = "Event_Nr"
        self.__annotationDataOverAll = dict()
        self.__normaltest_alpha = 0.05
        self.__normaltest_result_storage = dict()
        self.__normalTCloseness = True

        # Generate tree nodes and add event log data to them
        for index, row in eventLog.iterrows():
            activity = row[self.__activityColName]
            annotation = row[self.__annotationColName]
            if row[self.__caseIDColName] != currentCase:
                current = root
                if not sequence is None:
                    caseToSequenceDict[currentCase] = sequence
                    current.sequences.add(sequence)
                currentCase = row[self.__caseIDColName]
                current.cases.add(currentCase)
                sequence = ""
            childAlreadyExists = False
            sequence = sequence + "@" + activity
            for child in current.children:
                if child.name == activity:
                    childAlreadyExists = True
                    current = child
            if not childAlreadyExists:
                node = AnyNode(id=index, name=activity, parent=current, cases=set(), sequence=sequence, annotations=dict())
                current = node
            current.cases.add(currentCase)
            current.annotations[currentCase] = annotation
            self.__addAnnotation(annotation, activity)
        #Handle last case
        caseToSequenceDict[currentCase] = sequence
        root.sequences.add(sequence)
        self._tree = root
        self._caseToSequenceDict = caseToSequenceDict
        self.__numberOfTracesOriginal = len(self._tree.cases)
        self._sequentialPrunning = True
        self.__setMaxDifferences()
        self.__haveAllValuesInActivitityDistributionTheSameValue = dict()
        self._distanceMatrix = self.__generateDistanceMatrixSequences(self._getAllPotentialSequencesTree(self._tree))

    def __addAnnotation(self, annotation, activity):
        dataForActivity = self.__annotationDataOverAll.get(activity, None)
        if dataForActivity is None:
            self.__annotationDataOverAll[activity] = []
            dataForActivity = self.__annotationDataOverAll[activity]
        dataForActivity.append(annotation)

    def __setMaxDifferences(self):
        self.annotationMaxDifferences = dict()
        for key in self.__annotationDataOverAll.keys():
            maxVal = max(self.__annotationDataOverAll[key])
            minVal = min(self.__annotationDataOverAll[key])
            self.annotationMaxDifferences[key] = abs(maxVal - minVal)

    def _violatesTCloseness(self, activity, annotations, t, cases):
        distributionActivity = self.__annotationDataOverAll[activity]
        maxDifference = self.annotationMaxDifferences[activity]
        #Consider only data from cases still in node
        distributionEquivalenceClass = []
        casesInClass = cases.intersection(set(annotations.keys()))
        for caseInClass in casesInClass:
            distributionEquivalenceClass.append(annotations[caseInClass])
        if len(distributionEquivalenceClass) == 0: #No original annotation is left in the node
            return False
        if maxDifference == 0.0: #All annotations have the same value(most likely= 0.0)
            return
        if self.__normalTCloseness == True:
            return ((wasserstein_distance(distributionActivity,distributionEquivalenceClass)/maxDifference) >= t)
        else:
            return self._violatesStochasticTCloseness(distributionActivity,distributionEquivalenceClass,t,activity)


    def _cutCasesOutOfTreeStartingFromNode(self,node,cutOutTraces,tree=None):
        if tree == None:
            tree = self._tree
        current = node
        try:
            tree.sequences.remove(node.sequence)
        except KeyError:
            pass
        while current != tree:
            current.cases = current.cases.difference(cutOutTraces)
            if len(current.cases) == 0:
                node = current
                current = current.parent
                node.parent = None
            else:
                current = current.parent

    def _getAllPotentialSequencesTree(self, tree):
        return tree.sequences

    def _addCaseToTree(self, trace, sequence,tree=None):
        if tree == None:
            tree = self._tree
        if trace != "":
            activities = sequence.split("@")
            currentNode = tree
            tree.cases.add(trace)
            for activity in activities:
                for child in currentNode.children:
                    if child.name == activity:
                        child.cases.add(trace)
                        currentNode = child
                        break

    def __combineTracesAndTree(self, traces):
        #We transform the set of sequences into a list and sort it, to discretize the behaviour of the algorithm
        sequencesTree = list(self._getAllPotentialSequencesTree(self._tree))
        sequencesTree.sort()
        for trace in traces:
            bestSequence = ""
            #initial value as high as possible
            lowestDistance = sys.maxsize
            traceSequence = self._caseToSequenceDict[trace]
            for treeSequence in sequencesTree:
                currentDistance = self._getDistanceSequences(traceSequence, treeSequence)
                if currentDistance < lowestDistance:
                    bestSequence = treeSequence
                    lowestDistance = currentDistance
            self._overallLogDistance += lowestDistance
            self._addCaseToTree(trace, bestSequence)

    def __generateNewAnnotation(self, activity):
        #normaltest works only with more than 8 samples
        if(len(self.__annotationDataOverAll[activity])) >=8 and activity not in self.__normaltest_result_storage.keys():
            stat, p = normaltest(self.__annotationDataOverAll[activity])
        else:
            p = 1.0
        self.__normaltest_result_storage[activity] = p
        if self.__normaltest_result_storage[activity] <= self.__normaltest_alpha:
            mean = np.mean(self.__annotationDataOverAll[activity])
            std = np.std(self.__annotationDataOverAll[activity])
            randomValue = np.random.normal(mean, std)
        else:
            randomValue = np.random.choice(self.__annotationDataOverAll[activity])
        if randomValue < 0:
            randomValue = 0
        return randomValue

    def getEvent(self,case,node):
        event = {
            self.__activityColName: node.name,
            self.__caseIDColName: case,
            self.__annotationColName: node.annotations.get(case, self.__generateNewAnnotation(node.name)),
            self.__constantEventNr: node.depth
        }
        return event

    def getEventsOfNode(self, node):
        events = []
        if node != self._tree:
            events = events + [self.getEvent(case, node) for case in node.cases]
        return events

    def __generateDistanceMatrixSequences(self,sequences):
        distanceMatrix = dict()
        for sequence1 in sequences:
            distanceMatrix[sequence1] = dict()
            for sequence2 in sequences:
                if sequence1 != sequence2:
                    distanceMatrix[sequence1][sequence2] = levenshtein(sequence1,sequence2)
        print("Generated Distance Matrix")
        return distanceMatrix

    def _getDistanceSequences(self, sequence1, sequence2):
        if sequence1 == "" or sequence2 == "" or sequence1 == sequence2:
            return sys.maxsize
        try:
            distance = self._distanceMatrix[sequence1][sequence2]
        except KeyError:
            print("A Sequence is not in the distance matrix")
            print(sequence1)
            print(sequence2)
            raise
        return distance

    def __areAllValuesInDistributionAreTheSame(self, distribution):
        if max(distribution) == min(distribution):
            return True
        else:
            return False

    def _violatesStochasticTCloseness(self,distributionEquivalenceClass,overallDistribution,t,activity):
        if activity not in self.__haveAllValuesInActivitityDistributionTheSameValue.keys():
            self.__haveAllValuesInActivitityDistributionTheSameValue[activity] = self.__areAllValuesInDistributionAreTheSame(overallDistribution)
        if not self.__haveAllValuesInActivitityDistributionTheSameValue[activity]:
            upperLimitsBuckets = self._getBucketLimits(t,overallDistribution)
            return (self._calculateStochasticTCloseness(overallDistribution, distributionEquivalenceClass, upperLimitsBuckets) > t)
        else:
            return False

    def _calculateStochasticTCloseness(self, overallDistribution, equivalenceClassDistribution, upperLimitBuckets):
        overallDistribution.sort()
        equivalenceClassDistribution.sort()
        counterOverallDistribution = 0
        counterEquivalenceClass = 0
        distances = list()
        for bucket in upperLimitBuckets:
            lastCounterOverallDistribution = counterOverallDistribution
            lastCounterEquivalenceClass = counterEquivalenceClass
            while counterOverallDistribution<len(overallDistribution) and overallDistribution[counterOverallDistribution
            ] < bucket:
                counterOverallDistribution = counterOverallDistribution + 1
            while counterEquivalenceClass<len(equivalenceClassDistribution) and equivalenceClassDistribution[counterEquivalenceClass
            ] < bucket:
                counterEquivalenceClass = counterEquivalenceClass + 1
            probabilityOfBucketInEQ = (counterEquivalenceClass-lastCounterEquivalenceClass)/len(equivalenceClassDistribution)
            probabilityOfBucketInOverallDistribution = (counterOverallDistribution-lastCounterOverallDistribution)/len(overallDistribution)
            if probabilityOfBucketInEQ == 0 and probabilityOfBucketInOverallDistribution == 0:
                distances.append(0)
            elif probabilityOfBucketInOverallDistribution == 0 or probabilityOfBucketInEQ == 0:
                distances.append(sys.maxsize)
            else:
                distances.append(max(probabilityOfBucketInEQ/probabilityOfBucketInOverallDistribution,probabilityOfBucketInOverallDistribution/probabilityOfBucketInEQ))
        return max(distances)



    def _getBucketLimits(self,t,overallDistribution):
        numberOfBuckets = round(t+1)
        overallDistribution.sort()
        divider = round(len(overallDistribution)/numberOfBuckets)
        upperLimitsBuckets = list()
        for i in range(1,numberOfBuckets):
            upperLimitsBuckets.append(overallDistribution[min(round(i*divider),len(overallDistribution)-1)])
        return upperLimitsBuckets

    def getPrivatisedEventLog(self, applydp, epsilon=0.5):
        """
        Retrieves a privatized event log.

        Args:
            applydp (bool): Whether to apply differential privacy.
            epsilon (float): Epsilon value for differential privacy (default is 0.5).

        Returns:
            pd.DataFrame: Privatized event log.

        This function retrieves an event log and optionally applies differential privacy.
        """
        events = []
        self.__normaltest_result_storage = dict()
        nodeEvents = [self.getEventsOfNode(node) for node in PreOrderIter(self._tree)]
        for node in nodeEvents:
            events.extend(node)
        eventLog = pd.DataFrame(events)
        if not eventLog.empty:
            eventLog = eventLog.sort_values(by=[self.__caseIDColName, self.__constantEventNr])
        # checks if differential privacy is required
        if applydp:
            # Set up the differential mechanism based on the original dataset
            DPmechanisms = self.__DPmechanismSetup(epsilon)
            print("Differential privacy mechanism ready")
            # Apply differential privacy to annotation
            self.__applyDiffPrivacy(DPmechanisms, eventLog)
            print(f"Differential privacy applied to event log epsilon={epsilon}")
        # return a copy of the sanitized event log
        return eventLog.copy()

    def runPretsa(self, k, t, l, normalTCloseness=True):
        """
        Run PRETSA algorithm with optional parameters.

        Args:
            k (int): K-anonymity parameter.
            t (float): T-closeness parameter.
            l (int): L-diversity parameter.
            normalTCloseness (bool): Whether to use normal T-closeness (default is True).

        Returns:
            tuple: A tuple containing cutOutCases and the overallLogDistance.

        This function runs the PRETSA algorithm with specified parameters.
        """
        self.__normalTCloseness = normalTCloseness
        if not self.__normalTCloseness:
            self.__haveAllValuesInActivitityDistributionTheSameValue = dict()
        self._overallLogDistance = 0.0
        if self._sequentialPrunning:
            cutOutCases = set()
            # Initial tree pruning
            cutOutCase = self._treePrunning(k, t, l)
            # Keep pruning until no change is observed
            while len(cutOutCase) > 0:
                self.__combineTracesAndTree(cutOutCase)
                cutOutCases = cutOutCases.union(cutOutCase)
                cutOutCase = self._treePrunning(k, t, l)
        else:
            cutOutCases = self._treePrunning(k, t, l)
            self.__combineTracesAndTree(cutOutCases)
        return cutOutCases, self._overallLogDistance

    def _treePrunning(self, k, t, l):
        """
        Perform tree pruning based on specified parameters.

        Args:
            k (int): K-anonymity parameter.
            t (float): T-closeness parameter.
            l (int): L-diversity parameter.

        Returns:
            set: Set of cut-out traces.

        This function performs tree pruning based on specified parameters and returns a set of cut-out traces.
        """
        cutOutTraces = set()
        for node in PreOrderIter(self._tree):
            if node != self._tree:
                node.cases = node.cases.difference(cutOutTraces)
                # checks that each node satisfy k-anonymity, l-diversity, and t-closeness
                if (len(node.cases) < k or self._violatesTCloseness(node.name, node.annotations, t, node.cases)
                        or self.__violatesLDiversity(node.name, node.annotations, l, node.cases)):
                    cutOutTraces = cutOutTraces.union(node.cases)
                    self._cutCasesOutOfTreeStartingFromNode(node, cutOutTraces)
                    if self._sequentialPrunning:
                        return cutOutTraces
        return cutOutTraces

    def __applyDiffPrivacy(self, DPmechanisms, sanitizedEventlog):
        """
        Apply differential privacy to the anonymized event log.

        Args:
            DPmechanisms (dict): Dictionary of differential privacy mechanisms.
            sanitizedEventlog (pd.DataFrame): Anonymized event log generated from the PRETSA algorithm.

        This function applies differential privacy to the annotation in the anonymized event log.
        """
        # add calculated and bounded noise to each row's sensitive attribute
        for index, row in sanitizedEventlog.iterrows():
            sanitizedEventlog.at[index, self.__annotationColName] = \
                DPmechanisms[row['Activity']].randomise(sanitizedEventlog.at[index, self.__annotationColName])

    def __DPmechanismSetup(self, epsilon):
        """
        Set up differential privacy mechanisms.

        Args:
            epsilon (float): Epsilon value for differential privacy.

        Returns:
            dict: Dictionary of differential privacy mechanisms.

        This function sets up differential privacy mechanisms and returns them as a dictionary.
        """
        numericalMech = self.__initNumMech(epsilon)
        return numericalMech

    def __initNumMech(self, epsilon):
        """
        Initialize numerical mechanisms.

        Args:
            epsilon (float): Privacy budget for differential privacy.

        Returns:
            dict: Dictionary of numerical mechanisms for each activity in the event log.

        This function initializes numerical mechanisms for differential privacy and returns them as a dictionary.
        """
        numericalMechanism = dict()
        df = self.__eventlog
        # list of activities in the event log
        listofActivities = df[self.__activityColName].unique()
        # for each activity calculate sensitivity, upper domain, and lower domain
        for activity in listofActivities:
            sameActivityEvents = df[df[self.__activityColName] == activity]
            upperDomain = sameActivityEvents[self.__annotationColName].max()
            lowerDomain = sameActivityEvents[self.__annotationColName].min()
            sensitivity = abs(upperDomain - lowerDomain)
            # add differential privacy mechanism to the dict
            numericalMechanism[activity] = dp.mechanisms.LaplaceBoundedDomain(epsilon=epsilon, upper=upperDomain,
                                                                              lower=lowerDomain,sensitivity=sensitivity)
        return numericalMechanism

    def __violatesLDiversity(self, activity, annotations, l, cases):
        """
        Check if L-diversity is violated.

        Args:
            activity (str): Activity name.
            annotations (dict): Annotations for cases.
            l (int): L-diversity parameter.
            cases (set): Set of cases.

        Returns:
            bool: True if L-diversity is violated; otherwise, False.

        This function checks if L-diversity is violated based on the specified parameters.
        """

        maxDifference = self.annotationMaxDifferences[activity]
        # All annotations have the same value (most likely = 0.0)
        if maxDifference == 0.0:
            return False
        # Consider only data from cases still in node
        equivalenceClass = []
        casesInClass = cases.intersection(set(annotations.keys()))
        for caseInClass in casesInClass:
            equivalenceClass.append(annotations[caseInClass])
        unique_values = set(equivalenceClass)
        # checks if each equivalence class has at least l entries
        # in other words, checks if the equivalence class satisfies k-anonymity needed to achieve l-diversity
        # if not return false
        if len(equivalenceClass) < l:
            return True
        # check if l-diversity is satisfied
        return len(unique_values) < l

