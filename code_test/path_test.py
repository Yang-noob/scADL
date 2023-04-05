import os

path1 = 'C:/Users/user1'
path2 = '/Desktop'
filename = '/example.txt'

full_path = os.path.join(path1, path2, filename)
print(full_path)  # 输出：C:/Users/user1/Desktop/example.txt
