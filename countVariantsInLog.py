def generateVariantColumn(event_log):
    current_case = ""
    current_variant = ""
    caseIDColName = "Case_ID"
    activityColName = "Activity"
    variants = []
    for index, row in event_log.iterrows():
        activity = row[activityColName]
        if row[caseIDColName] != current_case:
            current_variant = activity
            current_case = row[caseIDColName]
            variants.append(current_variant)
        else:
            current_variant = current_variant + "@" + activity
            variants.append(current_variant)
    # variants.add(current_variant)
    return variants

def count_variants(event_log,return_variants=False):
    current_case = ""
    current_variant = ""
    caseIDColName = "Case_ID"
    activityColName = "Activity"
    variants = set()
    for index, row in event_log.iterrows():
        activity = row[activityColName]
        if row[caseIDColName] != current_case:
            variants.add(current_variant)
            current_variant = ""
            current_case = row[caseIDColName]
        current_variant = current_variant + "@" + activity
    variants.add(current_variant)
    if return_variants:
        return len(variants) - 1, variants
    else:
        return len(variants) - 1

def testVariantGen(event_log):
    current_case = ""
    current_variant = ""
    caseIDColName = "Case_ID"
    activityColName = "Activity"
    variants = set()
    variantDict = dict()
    for index, row in event_log.iterrows():
        activity = row[activityColName]
        if row[caseIDColName] != current_case:
            if(len(current_variant) > 0):
                variantDict[current_case] = current_variant
                variants.add(current_variant)
            current_variant = ""
            current_case = row[caseIDColName]
        current_variant = current_variant + "@" + activity
    # handles adding the last iteration
    variants.add(current_variant)
    variantDict[current_case] = current_variant

    event_log['Variant'] = None
    for index, row in event_log.iterrows():
        event_log.at[index, 'Variant'] = variantDict[row['Case_ID']]
