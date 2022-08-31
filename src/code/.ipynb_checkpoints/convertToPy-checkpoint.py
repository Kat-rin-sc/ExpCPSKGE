jupyter nbconvert --to script KGE_runOnServer.ipynb
with open('KGE_runOnServer.py', 'r') as f:
    lines = f.readlines()
with open('KGE_runOnServer.py', 'w') as f:
    for line in lines:
        if 'nbconvert --to script' in line:
            break
        else:
            f.write(line)