import os

def find_the_way(path,file_format,con=""):
    files_add = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                if con in file:
                    files_add.append(os.path.join(r, file))  
            
    return files_add





def killthemall(path, uzanti):
    them=find_the_way(path, uzanti)
    for t in them:
        print(t)
        try:
            os.remove(t)
        except:
            print(f"error about delete {t} file")


path="./"
uzanti=".pcap"
killthemall(path, uzanti)
