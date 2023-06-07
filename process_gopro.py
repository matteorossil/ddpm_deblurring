import os

gopro_dirs = os.listdir()
i = 1

for dir in gopro_dirs:

    sharp_imgs = os.listdir(dir+"/train/sharp")
    blur_imgs = os.listdir(dir+"/train/blur")

    for s_img in sharp_imgs:
        os.rename(dir+"/train/sharp/"+s_img, dir+"/train/sharp/"+i+s_img)
    
    for b_img in blur_imgs:
        os.rename(dir+"/train/blur/"+s_img, dir+"/blur/sharp/"+i+s_img)

    i+=1



