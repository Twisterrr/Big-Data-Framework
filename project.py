from pyspark import SparkContext

# Conversion of the python function corr() to pyspark 
def calculate_correlation_matrix(rdd):
    # Collect the data from the dataframe into a list of lists
    data = rdd.collect()

    # Calculate the mean and standard deviation for each column
    means = [sum(row)/len(row) for row in zip(*data)]
    stds = [((sum([(row[i]-means[i])**2 for row in data]))/(len(data)-1))**0.5 for i in range(len(means))]

    # Calculate the Pearson correlation matrix
    corr_matrix = []
    for i in range(len(rdd.first())):
        corr_row = []
        for j in range(len(rdd.first())):
            if i == j:
                corr_row.append(1.0)
            else:
                corr = sum([(data[k][i]-means[i])*(data[k][j]-means[j]) for k in range(len(data))]) / ((len(data)-1)*stds[i]*stds[j])
                corr_row.append(corr)
        corr_matrix.append(corr_row)

    return corr_matrix

def printCorrelationMatrix(rdd):
    # Filter out rows with "null" and split each row by ","
    rdd = rdd.filter(lambda row: "null" not in row).map(lambda row: row.split(","))
    # Get the column names from the first row
    columns = rdd.first()
    # Extract the column names except for the first two columns
    columnNames = columns[2:]
    # Convert the data to float and remove the first two columns
    rdd = rdd.filter(lambda row: row != columns).map(lambda row: [float(val) for val in row[2:]])
    # If the number of rows in the dataframe is less than 2, there is no correlation matrix to show
    if rdd.count() < 2:
        print(f'No correlation matrix shown: The number of non-null or constant rows ({rdd.count()}) is less than 2')
        return
    # Calculate the correlation matrix
    corr = calculate_correlation_matrix(rdd)
    print(f'Correlation Matrix for Hapiness in 2015 : \n')
    for i, row in enumerate(corr):
        column_name = ["Happiness Rank", "Happiness Score", "Standard Error", "Economy (GDP per Capita)", "Family", "Health (Life Expectancy)", "Freedom", "Trust (Government Corruption)", "Generosity", "Dystopia Residual"]
        print(str(column_name[i]) + ":")
        print("\t".join(str(val) for val in row))
        print("\n")

def print_per_column_stats(rdd):
    # Filter columns with between 1 and 50 unique values
    nunique = rdd.map(lambda row: row.split(",")).map(lambda row: (row[0], row[1:])).mapValues(set).mapValues(len).collectAsMap()
    rdd = rdd.filter(lambda row: nunique[row.split(",")[0]] > 1 and nunique[row.split(",")[0]] < 50)

    # Calculation of the number of rows and columns of the dataframe
    nRow = rdd.count()
    columnNames = rdd.first().split(",")
    nCol = len(columnNames)
    print("Columns statistics :\n")

    # Iterate through each column of the dataframe and display its descriptive statistics
    for i, col in enumerate(columnNames[3:], start=3):
      column_name = ["Country", "Region", "Happiness Rank", "Happiness Score", "Standard Error", "Economy (GDP per Capita)", "Family", "Health (Life Expectancy)", "Freedom", "Trust (Government Corruption)", "Generosity", "Dystopia Residual"]
      print(str(column_name[i]) + ":")
      colData = rdd.map(lambda row: float(row.split(",")[i]) if row.split(",")[i] != "" else None).filter(lambda x: x is not None).collect()
      colData = sorted(colData)
      nData = len(colData)
      quartiles = [colData[int(nData * q)] for q in [0.25, 0.5, 0.75]]
      print(f"\tMinimum: {colData[0]}")
      print(f"\t25th percentile: {quartiles[0]}")
      print(f"\tMedian: {quartiles[1]}")
      print(f"\t75th percentile: {quartiles[2]}")
      print(f"\tMaximum: {colData[-1]}")
      print("\n")

def countCountriesByRegion(rdd):
    # Map each row to a tuple of (region, country)
    regions_countries = rdd.map(lambda row: row.split(',')).map(lambda cols: (cols[1]))
    # Define the 10 regions and their corresponding names
    regions = ["Western Europe", "North America", "Australia and New Zealand", "Middle East and Northern Africa",
               "Latin America and Caribbean", "Southeastern Asia", "Central and Eastern Europe", "Eastern Asia",
               "Sub-Saharan Africa", "Southern Asia"]
    # Initialize a dictionary to store the number of countries per region
    region_counts = dict.fromkeys(regions, 0)
    # Count the number of countries per region
    for region in regions_countries.collect():
        if region in regions:
            region_counts[region] += 1

    # Print the results as a table
    print("Histogram number of Countries by Regions : \n")
    print("Region\t\tNumber of Countries")
    print("-----------------------------------")
    for region, count in region_counts.items():
        print(f"{region}\t\t{count}")
    print("\n")
        
# Creating SparkContext
sc = SparkContext()
rdd = sc.textFile("./2015.csv")

# Filtering out the first line = header
header = rdd.first()
rdd = rdd.filter(lambda line: line != header)

# Display the number of rows and columns of the dataset
nRow = rdd.count()
nCol = header.split(',')
print(f'There are {nRow} rows and {len(nCol)} columns.\n')

# Display the first five rows
for row in rdd.take(5):
    print(row)
print("\n")

# Shows the number of countries in each region
countCountriesByRegion(rdd)

# Show descriptive statistics for each column of the dataframe
print_per_column_stats(rdd)

# Show descriptive correlation matrix
printCorrelationMatrix(rdd)

#Close the Spark session
sc.stop()
