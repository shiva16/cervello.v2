#import necessary modules
import csv
x = "III"
y = "MPO"
with open('bams.csv','rt')as f:
  data = csv.reader(f)
  for row in data:
        if (row[0]==x and row[1]==y):
        	print(f"BAMS score of the TERMS:",row[2])
            