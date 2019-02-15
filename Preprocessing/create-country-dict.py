# 1.) Import country list and parse into lines
country_list = open("../Data/countrylist.txt")
lines = country_list.readlines()

# 2.) Create {countryform:countrycode} dictionary
country_forms = {}

for line in lines:
    if line[0] == "=":
        continue

    split = line.split("\t")
    country_form = split[1]

    # remove '\n' char, remove ()'s, replace _ with space
    country_form = country_form.replace("\n", "")\
        .replace("(", "")\
        .replace(")", "")\
        .replace("_", " ")\

    country_forms[country_form] = split[0]



