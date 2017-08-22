#-*-coding:utf8;-*-
#qpy:3
#qpy:console


# --- File reading --- #


# Open file
file = open('../src/Positive/01.txt', encoding = 'utf-8')
data = file.read()
file.close()

# Output data
print('\n' + data + '\n')
