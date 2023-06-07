import os

gopro_dirs = os.listdir()
i = 1

for dir in gopro_dirs:

    sharp_imgs = os.listdir(dir+"/sharp")
    blur_imgs = os.listdir(dir+"/blur")

    for s_img in sharp_imgs:
        os.rename(dir+"/sharp/"+s_img, dir+"/sharp/"+str(i)+s_img)
    
    for b_img in blur_imgs:
        os.rename(dir+"/blur/"+b_img, dir+"/blur/"+str(i)+b_img)

    i+=1



