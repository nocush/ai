import pytube
import os

link = input("Podaj url do filmu: ")
yt = pytube.YouTube(link)
#tu masz do filmów
#ys = yt.streams.get_highest_resolution()
ys = yt.streams.filter(only_audio=True).first()
#zmien sobie sciezke pobierania
out_file = ys.download(r"C:\Users\matim\Downloads\movies")
#do muzyki
base, ext = os.path.splitext(out_file)
new_file = base + '.mp3'
os.rename(out_file,new_file)

#jednak działa
print("Downloaded: ",yt.title)