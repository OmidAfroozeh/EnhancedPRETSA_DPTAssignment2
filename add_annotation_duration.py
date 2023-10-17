import sys
import csv
import datetime

class excel_semicolon(csv.excel):
    delimiter = ';'

dataset = sys.argv[1]
filePath = sys.argv[2]

caseIdColName = "Case_ID"
durationColName = "Duration"

writeFilePath = filePath.replace(".csv","_duration.csv")
print(writeFilePath)

timeStampColName = "Complete_Timestamp"


with open(filePath) as csvfile:
    with open(writeFilePath,'w') as writeFile:
        reader = csv.DictReader(csvfile,delimiter=",")
        fieldNamesWrite = reader.fieldnames
        fieldNamesWrite.append(durationColName)
        writer = csv.DictWriter(writeFile, fieldnames=fieldNamesWrite,dialect=excel_semicolon)
        writer.writeheader()
        currentCase = ""
        timestampFormat = {'Sepsis':'%Y-%m-%d %H:%M:%S%z', 'CoSeLog':'%Y-%m-%d %H:%M:%S.%f%z', 'Traffic':''}
        for row in reader:
            if dataset != "bpic2017":
                newTimeStamp = datetime.datetime.strptime(row[timeStampColName], timestampFormat[dataset])
                if currentCase != row[caseIdColName]:
                    currentCase = row[caseIdColName]
                    duration = 0.0
                else:
                    duration = (newTimeStamp - oldTimeStamp).total_seconds()/86400
                oldTimeStamp = newTimeStamp
            else:
                startTimeStamp = datetime.datetime.strptime(row[timeStampColName], '%Y/%m/%d %H:%M:%S.%f')

                endTimeStamp = datetime.datetime.strptime(row[timeStampColName], '%Y/%m/%d %H:%M:%S.%f')
                duration = (endTimeStamp - startTimeStamp).total_seconds()/86400
            row[durationColName] = duration
            writer.writerow(row)
print(f"Duration notation added, Available at {writeFilePath}")