import re

string = "TCGA-AC-A23H-01Z-00-DX1_33376_10528.png"
ecoded = "_10036_10036.png"
# "_-68409.92504931378_-26349.29104672192.png"

# # Regular expression to match numbers
# pattern = r'\d+?\d*'

# # Find all numbers in the string
# numbers = re.findall(pattern, string)

# # Extract last two numbers
# last_two_numbers = numbers[-2:]

# # Print the last two numbers
# print(last_two_numbers)

def find_coordinates(patch_name): 
    # Regular expression to match numbers
    pattern = r'\d+?\d*'

    # Find all numbers in the string
    numbers = re.findall(pattern, patch_name)

    # Extract last two numbers
    last_two_numbers = numbers[-2:]
    if len(last_two_numbers) != 0: 
        # Print the last two numbers
        # print(last_two_numbers, len(last_two_numbers))
        # print(last_two_numbers[0], last_two_numbers[1])
        return (last_two_numbers[0], last_two_numbers[1])

def find_encrpytcoordinates(encryptpatch_name): 
    x = re.sub(r"\_+", " ", encryptpatch_name)
    y = re.sub(r"\.png*", " ", x)
    z = re.split(" ", y)
    
    numbers = []
    for i in z:
        if i !='':
            numbers.append(i)

    # Extract last two numbers
    last_two_numbers = numbers[-2:]
    if len(last_two_numbers) != 0: 
        return last_two_numbers[0], last_two_numbers[1]

def find_decrpytcoordinates(decryptpatch_name): 
    x = re.sub(r"\_+", " ", decryptpatch_name)
    y = re.sub(r"\.png*", " ", x)
    z = re.split(" ", y)
    
    numbers = []
    for i in z:
        if i !='':
            numbers.append(i)

    # Extract last two numbers
    last_two_numbers = numbers[-2:]
    if len(last_two_numbers) != 0: 
        return last_two_numbers[0], last_two_numbers[1]
    
# print(find_coordinates(string),find_coordinates(string)[0])
# print(find_encrpytcoordinates(ecoded),find_encrpytcoordinates(ecoded)[0])


