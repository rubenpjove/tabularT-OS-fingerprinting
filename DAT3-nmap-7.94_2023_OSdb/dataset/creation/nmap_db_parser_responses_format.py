# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
def string_to_hex (string):
    try:
        x = int(string, 16)
    except:
        ValueError("Hex value cannot be casted")
    return x

# %%
def hex_value (string):
    if "-" in string:
        values = string.split("-")
        if values[0]==values[1]:
            return string_to_hex(values[0])
        else:
            return (string_to_hex(values[0]),string_to_hex(values[1]))
        # return [string_to_hex(values[0]),string_to_hex(values[1])]
        # return random.randint(string_to_hex(values[0]),string_to_hex(values[1]))
    elif ">" in string : # FEW CASES
        return string_to_hex(string[1:])
        # return (">",string_to_hex(string[1:]))
        # return [string_to_hex(string[1:]),sys.maxsize]
        # return string_to_hex(string[1:])
        #return random.randint(string_to_hex(string[1:]),sys.maxsize)
    elif "<" in string: # NOT EXIST
        return ("<",string_to_hex(string[1:]))
        # return [-1,string_to_hex(string[1:])]
        # return string_to_hex(string[1:])
        # return random.randint(-1,string_to_hex(string[1:]))
    elif string == "":
        return -1
    else:
        return string_to_hex(string)

# %%
def TI_CI_II (string):
    if string == "Z":
        return ["Z",-1]
    elif string == "RD":
        return ["RD",-1]
    elif string == "RI":
        return ["RI",-1]
    elif string == "BI":
        return ["BI",-1]
    elif string == "I":
        return ["I",-1]
    elif string == "":
        return ["EM",-1]
    else:
        return ["NO",hex_value(string)]

# %%
def SS (string):
    if string == "S":
        return "S"
    elif string == "O":
        return "O"
    elif string == "":
        return "NO"
    else:
        raise ValueError("SS test does not contain a valid value")

# %%
def TS (string):
    if string == "U":
        return ["U",-1]
    elif string == "-1":
        return ["-1",-1]
    elif string == "":
        return ["EM",-1]
    else:
        return ["NO",hex_value(string)]

# %%
def O (string):
    if string == "":
        return "NO"
    else:
        result = []
        for letter in string:
            if letter == "L":
                result.append("L")
            if letter == "N":
                result.append("N")
            if letter == "M":
                result.append("M")
            if letter == "W":
                result.append("W")
            if letter == "T":
                result.append("T")
            if letter == "S":
                result.append("S")
        return result

# %%
def Y_N (string):
    if string == "Y":
        return "Y"
    elif string == "N":
        return "N"
    elif string == "":
        return "NO"
    else:
        raise ValueError("A test with Y/N response does not contain a valid value")

# %%
def CC (string):
    if string == "Y":
        return "Y"
    elif string == "N":
        return "N"
    elif string == "S":
        return "S"
    elif string == "O":
        return "O"
    elif string == "":
        return "NO"
    else:
        raise ValueError("A test with Y/N response does not contain a valid value")

# %%
def Q (string):
    response = ["NO","NO"]
    if "R" in string:
        response[0] = "R"
    if "U" in string:
        response[1] = "U"

    return response

# %%
def S (string):
    if string == "Z":
        return "Z"
    elif string == "A":
        return "A"
    elif string == "A+":
        return "A+"
    elif string == "O":
        return "O"
    elif string == "":
        return "NO"
    else:
        raise ValueError("The test S response does not contain a valid value")

# %%
def A (string):
    if string == "Z":
        return "Z"
    elif string == "S":
        return "S"
    elif string == "S+":
        return "S+"
    elif string == "O":
        return "O"
    elif string == "":
        return "NO"
    else:
        raise ValueError("The test A response does not contain a valid value")

# %%
def F (string):
    response = ["NO","NO","NO","NO","NO","NO","NO"]
    if "E" in string:
        response[0] = "E"
    if "U" in string:
        response[1] = "U"
    if "A" in string:
        response[2] = "A"
    if "P" in string:
        response[3] = "P"
    if "R" in string:
        response[4] = "R"
    if "S" in string:
        response[5] = "S"
    if "F" in string:
        response[6] = "F"

    return response

# %%
def RIPL_RID_RUCK (string):
    if string == "G":
        return ["G",-1]
    elif string == "":
        return ["EM",-1]
    else:
        return ["NO",hex_value(string)]

# %%
def RIPCK (string):
    if string == "G":
        return "G"
    elif string == "Z":
        return "Z"
    elif string == "I":
        return "I"
    elif string == "":
        return "NO"
    else:
        raise ValueError("The test RIPCK response does not contain a valid value")

# %%
def RUD (string):
    if string == "G":
        return "G"
    elif string == "I":
        return "I"
    elif string == "":
        return "NO"
    else:
        raise ValueError("The test RUD response does not contain a valid value")

# %%
def DFI (string):
    if string == "N":
        return "N"
    elif string == "S":
        return "S"
    elif string == "Y":
        return "Y"
    elif string == "O":
        return "O"
    elif string == "":
        return "NO"
    else:
        raise ValueError("The test DFI response does not contain a valid value")

# %%
def CD (string):
    if string == "Z":
        return ["Z",-1]
    elif string == "S":
        return ["S",-1]
    elif string == "O":
        return ["O",-1]
    elif string == "":
        return ["EM",-1]
    else:
        return ["NO",hex_value(string)]